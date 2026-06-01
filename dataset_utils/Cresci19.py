import tarfile
import json
import pandas as pd
from torch.utils.data import Dataset
import os
from constants import CLEAN_USER_DICT, USER_COLUMNS, merge_users_with_labels

class Cresci19(Dataset):
    """
        Dataset for the Cresci-2019. The zipped dataset file should be named 'cresci19-rtbust-2019.tar' and should be placed in the directory /datasets.
    """

    def __init__(self, root, extract_dir):

        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)

        with tarfile.open(root, "r:gz") as tar:
            tar.extractall(extract_dir)

        user_data_path = os.path.join(extract_dir, "cresci-rtbust-2019_tweets.json")
        labels_path = os.path.join(extract_dir, "cresci-rtbust-2019.tsv")

        assert os.path.exists(user_data_path)
        assert os.path.exists(labels_path)

        self.labels_dt = pd.read_csv(labels_path, sep="\t", header=None, names=["id", "label"])

        with open(user_data_path, "r") as f:
            data = json.load(f)

        normalized_data = pd.json_normalize(data)

        self.user_data = normalized_data.rename(columns={
            "user.id": "id",
            "user.name": "name",
            "user.screen_name": "screen_name",
    })

        self.user_data = self.user_data[[c for c in USER_COLUMNS if c in self.user_data.columns]]
        self.user_data = self.user_data.replace(CLEAN_USER_DICT)
        self.user_data = self.user_data.dropna(subset=USER_COLUMNS)
        self.user_data = self.user_data[self.user_data["lang"] == "en"]
        self.user_data = self.user_data.merge_users_with_labels(self.labels_dt, on="id", how="left")


    def __len__(self):
        return len(self.user_data)


    def __getitem__(self, idx):
        """
        retrieves a single user sample from the dataset

        :param idx: integer index of the user in the dataset
        :return: dictionary containing all user profile features for that index
        """
        profile = self.user_data.iloc[idx]
        return profile.to_dict()