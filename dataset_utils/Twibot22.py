from importlib.resources import path
import shutil
import json
import pandas as pd
from torch.utils.data import Dataset
import os
from collections import defaultdict
from warnings import warn
from constants import CLEAN_USER_DICT, USER_COLUMNS, merge_users_with_labels

class Twibot22(Dataset):
    """
    Dataset for the Twibot-22. The downloaded *.json files or a directory,named "Twibot22", containing them, need to be in the directory /datasets.
    """
    def __init__(self, root : str | None = None, sample_size_tweets : int | None = None):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
        """
        if root == None:
            root = "datasets"

        # creates subdirectory "Twibot22" if it doesn't exist 
        if not os.path.exists(os.path.join(root, "Twibot22")):
             os.makedirs(os.path.join(root, "Twibot22"), exist_ok=True)

        #moves all json files into the "Twibot22" subdirectory
        for file in os.listdir(root):
            if file.endswith(".json") | file.endswith(".csv"):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(root, "Twibot22", file)
                shutil.move(src_path, dst_path)

        #defines the paths to the user data, labels and tweets
        user_data_path = os.path.join(root, "Twibot22", "user.json")
        labels_path = os.path.join(root, "Twibot22", "label.csv")
        self.tweet_paths = [
            os.path.join(root, "Twibot22", f"tweet_{i}.json")
            for i in range(9)
        ]

        #validation of the expected paths 
        if not os.path.exists(user_data_path):
            warn(
                "Dataset 'Twibo22.user.json' file not found. Please make sure 'user.json' is in the dataset directory.",                    UserWarning
            )
            return

        if not os.path.exists(labels_path):
            warn(
                "Dataset 'Twibot22.label.tsv' file not found. Please make sure 'label.csv' is in the dataset directory.",                    UserWarning
            )
            return
        
        for path in self.tweet_paths:
            if not os.path.exists(path):
                warn(
                    f"Dataset 'Twibot20.{path}' file not found. Please make sure every file is donwloaded in the dataset directory." )
        print("100")      
    
        
        #reads the data from the files and cleans them
        self.labels_dt = pd.read_csv(labels_path)
        with open(user_data_path, "r") as f:
            user_data = json.load(f)

        normalized_data = pd.json_normalize(user_data, sep=".")

        # flatten the nested user object into the columns expected by the pipeline
        rename_map = {
            f"user.{column}": column
            for column in USER_COLUMNS
            if f"user.{column}" in normalized_data.columns
        }
        self.user_data = normalized_data.rename(columns=rename_map).copy()

        available_columns = [column for column in USER_COLUMNS if column in self.user_data.columns]
        self.user_data = self.user_data[available_columns]
        self.user_data = self.user_data.replace(CLEAN_USER_DICT)
        self.user_data = self.user_data.dropna(subset=available_columns)
        self.user_data = merge_users_with_labels(self.user_data, self.labels_dt)

        self.user_index = defaultdict(list)

        for path in self.tweet_paths:
            try:
                df = pd.read_json(path)
                for uid in df["user_id"].unique():
                    self.user_index[uid].append(path)
            except FileNotFoundError:
                continue

       

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
        paths = self.user_index.get(user_id, [])

        if not paths:
            return pd.DataFrame()

        dfs = []
        for path in paths:
            try:
                df = pd.read_json(path)
                dfs.append(df[df["user_id"] == user_id])
            except FileNotFoundError:
                continue

        return pd.concat(dfs, ignore_index=True)
    
    

if __name__ == "__main__":
    example = Twibot22()
    example.__getitem__(10)
    

        