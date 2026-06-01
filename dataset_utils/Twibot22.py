from importlib.resources import path
from logging import root
import shutil
import pandas as pd
from torch.utils.data import Dataset
import os

from dataset_utils.constants import CLEAN_USER_DICT, USER_COLUMNS, merge_users_with_labels

class Twibot22(Dataset):
    """"
    Dataset for the Twibot-22. The downloaded *.json files or a directory,named "Twibot22", containing them, need to be in the directory /datasets.
    """
    def __init__(self, root):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
        """
        # creates subdirectory "Twibot22" if it doesn't exist 
        if not os.path.exists(os.path.join(root, "Twibot22")):
             os.makedirs(os.path.join(root, "Twibot22"), exist_ok=True)

        #moves all json files into the "Twibot22" subdirectory
        for file in os.listdir(root):
            if file.endswith(".json"):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(root, "Twibot22", file)
                shutil.move(src_path, dst_path)

        #defines the paths to the user data, labels and tweets
        user_data_path = os.path.join(root, "Twibot22", "user.json")
        labels_path = os.path.join(root, "Twibot22", "label.csv")
        tweet_paths = [
            os.path.join(root, "Twibot22", f"tweet{i}.json")
            for i in range(9)
        ]

        #validation of the expected paths 
        assert os.path.exists(user_data_path)
        assert os.path.exists(labels_path)      
        for path in tweet_paths:
            assert os.path.exists(path)  

        #reads the data from the files and cleans them
        self.labels_dt = pd.read_csv(labels_path)
        self.user_data = pd.read_json(user_data_path, orient="records")

        self.user_data = self.user_data[USER_COLUMNS] 
        self.user_data = self.user_data.replace(CLEAN_USER_DICT)
        self.user_data = self.user_data.dropna(
        subset=['id',
        'name',
        'screen_name',
        'statuses_count',
        'followers_count',
        'friends_count',
        'favourites_count',
        'listed_count',
        'lang'])
        self.user_data = merge_users_with_labels(self.user_data, self.labels_dt)

    def __len__(self):
        """
        gives the amount of user profiles in the dataset
        :return: the length of the dataset
        """
        return len(self.user_data)
    
    def get_sample(self, idx):
        """
        reads out one user profile and its associated tweets from the dataset
        :param idx: the index for one sample of the dataset (user profile)
        :return: a dictionary with the user profile data, list of tweets.
        """
        # get row of the user profile, their id and load their tweets
        profile = self.user_data.iloc[idx]
        user_id = profile["id"]
        tweets = self.load_user_tweets(user_id)

        result = {
            "profile": profile.to_dict(),
            "tweets": tweets.to_dict("records"),
        }
        return result
    
    def load_user_tweets(self, user_id):
        """
        Helper function to lazily load tweets for a specific user.
        """
        all_tweets = []

        for path in self.tweet_paths:
            df = pd.read_json(path)

            tweets = df[df["user_id"] == user_id]

        if not tweets.empty:
            all_tweets.append(tweets)

        if len(all_tweets) == 0:
            return pd.DataFrame()

        return pd.concat(all_tweets, ignore_index=True)



        