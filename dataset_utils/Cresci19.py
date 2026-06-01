import tarfile
import json
import pandas as pd
from torch.utils.data import Dataset
import os
from constants import CLEAN_USER_DICT, USER_COLUMNS, merge_users_with_labels

class Cresci19(Dataset):
    """"
    Dataset for the Cresci-2019. The zipped dataset file should be named 'cresci-rtbust-2019.tar' and should be placed in the directory /datasets.
    """

    def __init__(self, root,extract_dir):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
        :param extract_dir: the directory where the dataset will be extracted.
        """
        #extract dataset into extract_dirif it is not already extracted
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            
            with tarfile.open(root, "r") as tar:
                tar.extractall(extract_dir)

        #validates the expected files are present in the extract_dir
        user_data_path = os.path.join(root,extract_dir, "cresci-rtbust-2019_tweets.json")
        labels_path = os.path.join(root,extract_dir, "cresci-rtbust-2019.tsv")

        assert os.path.exists(user_data_path)
        assert os.path.exists(labels_path)

        #reads the data from the files
        self.labels_dt = pd.read_csv(labels_path, sep="\t", header=None, names=["id", "label"])
        data = json.load(open(user_data_path, "r"))

        #normalizes the json format and renames the columns to match the expected format for the rest of the pipeline 
        normalized_data = pd.json_normalize(data)
        self.user_data = normalized_data.rename(columns={
        "user.id": "id",
        "user.id_str": "id_str",
        "user.name": "name",
        "user.screen_name": "screen_name",
        })      

        # merge into single self.user_data dataframe with only relevant columns
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

    #no real tweets are provided in the dataset, so we return an empty list for the tweets
    def __getitem__(self, idx):
        """
        reads out one user profile and its associated tweets from the dataset
        :param idx: the index for one sample of the dataset (user profile)
        :return:#no real tweets are provided in the dataset, so we return an empty list for the tweets

        """
       # get row if the user profile and its id
        #profile = self.user_data.iloc[idx]
        #user_id = profile["id"]
        """"
        # get all tweets from the selected user profile
        tweets = self.tweet_data[self.tweet_data["user_id"] == user_id]
        result = {
            "profile": profile.to_dict(),
            "tweets": tweets.to_dict("records"),
        }
        """
        return []