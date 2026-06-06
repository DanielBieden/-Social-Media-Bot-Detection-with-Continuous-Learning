import math
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def create_profile_vector(data):
    """
    creates the meta-data embedding vector for the provided user.
    :param data: a dictionary corresponding to one sample from the dataset. Should contain a 'profile' entry.
    :return: tensor of the meta-data embedding vector
    """
    # user profile embedding
    profile = data["profile"]
    profile_embedding = torch.tensor([
        len(profile["name"]),
        len(profile["screen_name"]),
        math.log(profile["statuses_count"]+1),
        math.log(profile["followers_count"]+1),
        math.log(profile["friends_count"]+1),
        math.log(profile["favourites_count"]+1),
        profile["protected"],
        profile["verified"],
    ])
    return profile_embedding

def create_tweet_vectors(data, tokenizer, model, batch_size = math.inf, max_tweets = math.inf, device ='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    creates the tweets embedding vector for the provided user.
    :param device: the device on which to process the data.
    :param model: the model used for creating semantic embeddings from tweets
    :param tokenizer: the tokenizer used for creating tokens from text strings
    :param data: a dictionary corresponding to one sample from the dataset. Should contain a 'tweets' entry.
    :param batch_size: the batch size used to process the tweets. Defaults to processing all tweets at once
    :param max_tweets: the maximum amount of tweets to process per profile. Defaults to processing all available tweets at once.
    :return: tensor of the tweets embedding vector
    """
    tweet_count = len(data["tweets"])
    # if no tweets are available, return a default vector
    if tweet_count == 0:
        return torch.atleast_2d(torch.zeros(768)).detach().cpu()

    # get the maximum wanted tweets to process
    texts = list(map(lambda x: x["text"],data["tweets"]))[0: min(tweet_count, max_tweets)]
    # process texts in batches
    max_embedding = None
    raw_embeddings = []
    index = 0
    while index < min(tweet_count, max_tweets):
        # grab text batch
        text_batch = texts[index:min(index+batch_size, tweet_count)]
        # replace urls with 'url'
        URL_PATTERN = r"https?://\S+|www\.\S+"
        text_batch = [re.sub(URL_PATTERN, " url ", text) for text in text_batch]

        # get textual embedding of the batch
        with torch.no_grad():
            tokenized_text = tokenizer(text_batch, padding="longest", truncation=True, return_tensors="pt").to(device)
            output = model(**tokenized_text)
        # collect embeddings
        raw_embeddings.extend(torch.unbind(output.last_hidden_state[:, 0, :].detach().cpu(),dim=0))
        final_embeddings = torch.stack(raw_embeddings, dim=0)

        index += batch_size

    return torch.atleast_2d(final_embeddings).detach().cpu()


def train_classifier(classifier, criterion, optimizer ,input_samples, ground_truth_labels, epochs = 15, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    classifier.train()
    # create dataset from ground_truths and embedding features for batch processing
    input_tensors = torch.tensor(np.array(input_samples))
    ground_truth_tensors = torch.tensor(ground_truth_labels)
    dataset = TensorDataset(input_tensors, ground_truth_tensors)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0

        predictions = []
        ground_truths = []

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.float().to(device)
            targets = targets.to(device)

            # pass data through classifier and calculate loss
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)
            # optimize classifier
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # prediction
            prediction = torch.argmax(outputs, dim=1)

            # statistics
            predictions.extend(prediction.detach().cpu().numpy())
            ground_truths.extend(targets.detach().cpu().numpy())
            running_loss += loss.item() * inputs.size(0)

        # Calculate average loss across the entire epoch
        epoch_loss = running_loss / len(input_samples)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}")
    return ground_truths, predictions

def test_classifier(classifier, input_samples, ground_truth_labels, device = 'cuda' if torch.cuda.is_available() else 'cpu'):

    # create dataset from ground_truths and embedding features for batch processing
    input_tensors = torch.tensor(np.array(input_samples))
    ground_truth_tensors = torch.tensor(ground_truth_labels)
    dataset = TensorDataset(input_tensors, ground_truth_tensors)
    test_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    classifier.eval()

    predictions = []
    ground_truths = []

    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.float().to(device)
        targets = targets.to(device)

        # pass data through classifier and calculate loss
        with torch.inference_mode():
            outputs = classifier(inputs)

        # prediction
        prediction = torch.argmax(outputs, dim=1)

        predictions.extend(prediction.detach().cpu().numpy())
        ground_truths.extend(targets.detach().cpu().numpy())
    return ground_truths, predictions