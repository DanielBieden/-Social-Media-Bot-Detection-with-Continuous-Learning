import math
import re
import torch


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

def create_tweet_vector(data, tokenizer, model ,batch_size = math.inf, max_tweets = math.inf, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
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