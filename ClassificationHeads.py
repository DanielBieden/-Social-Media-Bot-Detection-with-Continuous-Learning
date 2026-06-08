import torch
from torch import nn
from triton.language import argmax


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
    On expansion a parallel classification head is added to accommodate new classes.
    The old classification layers are removed from further optimization once a new layer is added
    """
    def __init__(self, in_features, output_dim, dropout_p = 0.1):
        super().__init__()
        self.in_features = in_features
        self.num_classes = output_dim
        self.device = None # needed so that new layers are on the correct device

        self.dropout = nn.Dropout(p=dropout_p)
        # contains all output layers for classification
        self.classificationLayers = nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=output_dim)
        ])

    def forward(self, x):
        # set the device once
        if self.device is None: self.device = x.device

        x = self.dropout(x)
        # pass through all classification layers in parallel
        outputs = list(map(lambda layer: layer(x), self.classificationLayers))
        outputs = torch.cat(outputs, dim=-1) # concat outputs
        return outputs

    def expand_classifier(self, num_to_add = 1):
        """
        dynamically expands the classifier by the number of classes specified.
        Weights for the old classes are removed from the computation graph that is responsible for optimization
         and can't be trained anymore
        :param num_to_add: the number of classes to add
        :return: the new relevant parameters of the model to optimize
        """
        # disable optimization for old layers
        for layer in self.classificationLayers:
            for param in layer.parameters():
                param.requires_grad = False
        # add new layer to list
        new_layer = nn.Linear(self.in_features, num_to_add).to(device=self.device)
        self.classificationLayers.append(new_layer)
        return new_layer.parameters()