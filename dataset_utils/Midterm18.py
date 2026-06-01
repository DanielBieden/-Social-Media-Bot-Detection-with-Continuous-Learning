import tarfile
import json
import pandas as pd
from torch.utils.data import Dataset
import os
from constants import CLEAN_USER_DICT, USER_COLUMNS, merge_users_with_labels

class Midterm18(Dataset):
    """"
    Dataset for the Midterm-2018. The zipped dataset file should be named 'midterm-2018.tar' and should be placed in the directory /datasets.
    """

    def __init__(self, root, extract_dir):
        """ :param root: the filepath of the dataset directory in which the dataset is stored. 
            :param extract_dir: the directory where the dataset will be extracted. 
        """
        #extract dataset into extract_dir if it is not already extracted
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)

            with tarfile.open(root, "r:gz") as tar:
                tar.extractall(extract_dir)

        #validates the expected files are present in the extract_dir
        user_data_path = os.path.join(
            extract_dir,
            "midterm-2018_processed_user_objects.json"
        )

        labels_path = os.path.join(
            extract_dir,
            "midterm-2018.tsv"
        )

        assert os.path.exists(user_data_path)
        assert os.path.exists(labels_path)

        #reads the data from the files
        self.labels_dt = pd.read_csv(
            labels_path,
            sep="\t",
            header=None,
            names=["id", "label"]
        )

        with open(user_data_path, "r") as f:
            data = json.load(f)

        normalized = pd.json_normalize(data)

        # cleans the data
        self.user_data = normalized.rename(columns={
            "user.id": "id",
            "user.name": "name",
            "user.screen_name": "screen_name",
        })

        self.user_data = self.user_data[
            [c for c in USER_COLUMNS if c in self.user_data.columns]
        ]

        self.user_data = self.user_data.replace(CLEAN_USER_DICT)
        self.user_data = self.user_data.dropna(subset=USER_COLUMNS)

        self.user_data = self.user_data[self.user_data["lang"] == "en"]

        #merges users with labels to have a single dataframe with all relevant information for the rest of the pipeline
        self.user_data = self.user_data.merge_users_with_labels(
            self.labels_dt,
            on="id",
            how="left"
        )

    def __len__(self):
        """
        gives the amount of user profiles in the dataset
        :return: the length of the dataset
        """
        return len(self.user_data)

    def __getitem__(self, idx):

        profile = self.user_data.iloc[idx]
        return profile.to_dict()