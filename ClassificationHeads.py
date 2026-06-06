import torch
from torch import nn


class BaselineClassificationHead(nn.Module):
    """
    The baseline classification head is essentially a normal Neural Network without any continuous learning aspects.
    It is used to determine the upper bound of our backbone.
    """
    def __init__(self, embedding_dim, output_dim, dropout_prob = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(in_features=embedding_dim, out_features=output_dim)
    def forward(self, x):
        x = self.dropout(x)
        x = self.classifier(x)
        return x



class ProgressiveNeuralNetworkClassifier(nn.Module):
    """
    A simple progressive neural network that expands in size to accommodate new classes.
    On expansion the weights responsible for the old classes are frozen and don't change with gradient updates
    """
    def __init__(self, in_features, output_dim, dropout_p = 0.1):
        super().__init__()
        self.in_features = in_features
        self.num_classes = output_dim

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(in_features=in_features, out_features=output_dim)
        # add backpropagation hook to modify gradients during backpropagation
        self.fc.weight.register_hook(self._weight_freeze_hook)
        if self.fc.bias is not None:
            self.fc.bias.register_hook(self._weight_freeze_hook)

        # contains the indices of classes to have frozen
        self.frozenLayers = []

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _weight_freeze_hook(self, grad):
        """
        the function should be added as a hook during backpropagation. It ensures that gradients for frozen classes are not modified
        by setting the gradients to 0.
        :param grad: gradients for the gradient update
        :return: the modified gradients
        """
        # zero out unwanted gradient updates in order to not change gradients
        grad_clone = grad.clone()
        grad_clone[self.frozenLayers] = 0
        #print("gradient: " ,grad_clone)
        return grad_clone

    def expand_classifier(self, num_to_add = 1):
        """
        dynamically expands the classifier by the number of classes specified.
        Weights for the old classes are frozen and can't be trained anymore
        :param num_to_add: the number of classes to add
        """
        self.frozenLayers = list(range(0,self.num_classes)) # update list to contain old classes
        old_fc = self.fc
        old_num_classes = self.num_classes
        self.num_classes += num_to_add

        # create new dense layer with added output size
        new_fc = nn.Linear(self.in_features, self.num_classes)
        new_fc = new_fc.to(device=old_fc.weight.device, dtype=old_fc.weight.dtype)

        # copy the old weights and biases into the new layer (unpopulated weights are random)
        # add backpropagation hook to layer weights, so that freezing works
        with torch.no_grad():
            new_fc.weight[:old_num_classes, :] = old_fc.weight
            new_fc.weight.register_hook(self._weight_freeze_hook)
            if old_fc.bias is not None:
                new_fc.bias[:old_num_classes] = old_fc.bias
                new_fc.bias.register_hook(self._weight_freeze_hook)

        # replace layer
        self.fc = new_fc


