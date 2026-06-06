from enum import Enum
from zipfile import ZipFile
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
from constants import CLEAN_USER_DICT


class Cresci17SetTypes(Enum):
    GENUINE_USER = 1    
    FAKE_FOLLOWER = 2
    SOCIAL_SPAM_1 = 3
    SOCIAL_SPAM_2 = 4
    SOCIAL_SPAM_3 = 5
    TRADITIONAL_SPAM_1 = 6
    TRADITIONAL_SPAM_2 = 7
    TRADITIONAL_SPAM_3 = 8
    TRADITIONAL_SPAM_4 = 9


class Cresci17(Dataset):
    """
    Dataset for the Cresci-2017. The zipped dataset file should be named 'cresci-2017.csv.zip' and should be placed in the directory /datasets.
    """
    def __init__(self, root, subset_type):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
        :param subset_type: the wanted subset of the dataset.
        """
        self.subset_type = subset_type

        # select appropriate subdirectory to find wanted subset
        default_file_path = os.path.join(root, "cresci-2017" ,"datasets_full.csv")
        subset_path  = ""
        match subset_type:
            case Cresci17SetTypes.GENUINE_USER:
                subset_path = "genuine_accounts.csv"
            case Cresci17SetTypes.FAKE_FOLLOWER:
                subset_path = "fake_followers.csv"
            case Cresci17SetTypes.SOCIAL_SPAM_1:
                subset_path = "social_spambots_1.csv"
            case Cresci17SetTypes.SOCIAL_SPAM_2:
                subset_path = "social_spambots_2.csv"
            case Cresci17SetTypes.SOCIAL_SPAM_3:
                subset_path = "social_spambots_3.csv"
            case Cresci17SetTypes.TRADITIONAL_SPAM_1:
                subset_path = "traditional_spambots_1.csv"
            case Cresci17SetTypes.TRADITIONAL_SPAM_2:
                subset_path = "traditional_spambots_2.csv"
            case Cresci17SetTypes.TRADITIONAL_SPAM_3:
                subset_path = "traditional_spambots_3.csv"
            case Cresci17SetTypes.TRADITIONAL_SPAM_4:
                subset_path = "traditional_spambots_4.csv"
            case _:
                raise ValueError(f"Unknown subset type for Cresci17 dataset was used: {subset_type}")


        user_data_path = os.path.join(default_file_path, subset_path, "users.csv")
        tweet_data_path = os.path.join(default_file_path, subset_path, "tweets.csv")

        # automatic unzipping of files if they don't exist in an unpackaged state
        # check if files exist
        if not os.path.exists(user_data_path) or not os.path.exists(tweet_data_path):
            # check if dataset in whole exists, else unzip it
            if not os.path.exists(default_file_path):
                # check if zip file for the whole dataset exists
                if not os.path.exists(os.path.join(root, "cresci-2017.csv.zip")):
                    raise FileNotFoundError(f"no dataset structure or dataset zip file 'cresci-2017.csv.zip' found' at {root}")
                # unzip whole dataset
                with ZipFile(os.path.join(root, "cresci-2017.csv.zip")) as zipObj:
                    os.mkdir(os.path.join(root, "cresci-2017"))
                    zipObj.extractall(os.path.join(root, "cresci-2017"))
            # unzip subset
            with ZipFile(os.path.join(default_file_path, subset_path + ".zip")) as zipObj:
                zipObj.extractall(default_file_path)
                zipObj.close()

        # load subset dataset with only relevant entries
        self.user_data = pd.read_csv(user_data_path, usecols=['id','name','screen_name','statuses_count','followers_count','friends_count',
                                                             'favourites_count','listed_count','lang','protected','verified'])
        self.tweet_data = pd.read_csv(tweet_data_path, encoding="latin-1", dtype={'id': str},
                                      usecols=['id', 'text', 'user_id', 'in_reply_to_status_id', 'in_reply_to_user_id', 'in_reply_to_screen_name',
                                        'retweet_count', 'reply_count', 'favorite_count', 'num_hashtags', 'num_urls', 'num_mentions', 'timestamp'])

        # clean up missing values
        self.user_data = self.user_data.replace(CLEAN_USER_DICT)
        clean_dict = {
            'favourites_count':{np.nan: 0},
        }
        self.tweet_data = self.tweet_data.replace(clean_dict)

        # remove tweets and user profiles without text content
        self.tweet_data = self.tweet_data.dropna(subset=['text'])
        self.user_data = self.user_data.dropna(subset=['name', 'screen_name'])

        #print("medians:", self.user_data.median())

    def __len__(self):
        """
        gives the amount of user profiles in the dataset
        :return: the length of the dataset
        """
        return len(self.user_data)

    def __getitem__(self, idx):
        """
        reads out one user profile and its associated tweets from the dataset
        :param idx: the index for one sample of the dataset (user profile)
        :return: a dictionary with the user profile data, list of tweets. Additionally, returns the label for the dataset
        """
        # get row of the user profile and their id
        profile = self.user_data.iloc[idx]
        user_id = profile["id"]

        # get all tweets from the selected user profile
        tweets = self.tweet_data[self.tweet_data["user_id"] == user_id]
        result = {
            "profile": profile.to_dict(),
            "tweets": tweets.to_dict("records"),
        }
        return result, self.subset_type.value
