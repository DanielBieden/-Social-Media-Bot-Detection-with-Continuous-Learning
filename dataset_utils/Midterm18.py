from logging import root
import tarfile
import json
from warnings import warn
import pandas as pd
from torch.utils.data import Dataset
import os
from constants import CLEAN_USER_DICT, USER_COLUMNS, merge_users_with_labels

class Midterm18(Dataset):
    """"
    Dataset for the Midterm-2018. The zipped dataset file should be named 'midterm-2018.tar.gz' and should be placed in the directory /datasets.
    """

    def __init__(self, root : str):
        """ 
        self.user_data will contain all user profile features and the label for each user in the dataset. 
        :param root: the filepath of the dataset directory in which the dataset is stored. 
        """

        if not os.path.exists(root):
            warn(f"Dataset directory not found. Please make sure the dataset directory exists at the specified path: {root}")
            return
        
        #extract dataset into extract_dir if it is not already extracted
        default_path = os.path.join(root, "midterm-2018.tar.gz")
        extract_dir = os.path.join(root, "Midterm18")

        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(default_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            os.remove(default_path) 
            print("Finished 'Midterm18' extraction and removed the zip file")

        #validates the expected files are present in the extract_dir
        user_data_path = os.path.join(extract_dir, "midterm-2018_processed_user_objects.json")
        labels_path = os.path.join(extract_dir, "midterm-2018.tsv")

        if not os.path.exists(user_data_path):
            warn(
                "Dataset 'Midterm18.midterm-2018_processed_user_objects.json' file not found. Please make sure 'midterm-2018.tar.gz' is in the dataset directory and that it contains the expected files.",                    UserWarning
            )
            return
        
        if not os.path.exists(labels_path):
            warn(
                "Dataset 'Midterm18.midterm-2018.tsv' file not found. Please make sure 'midterm-2018.tar.gz' is in the dataset directory and that it contains the expected files.",                    UserWarning
            )
            return

        #reads the data and normalizes it into the right format  
        self.labels_dt = pd.read_csv(
            labels_path,
            sep="\t",
            header=None,
            names=["id", "label"]
        )

        with open(user_data_path, "r") as f:
            data = json.load(f)
       
        normalized_data = pd.json_normalize(data)

        self.user_data = normalized_data.rename(columns={"user_id": "id"}).copy()

        available_columns = [column for column in USER_COLUMNS if column in self.user_data.columns]
        self.user_data = self.user_data[available_columns]
        self.user_data = self.user_data.dropna(subset=available_columns)
        self.user_data = self.user_data[self.user_data["lang"] == "en"]

        #merges users with labels to have a single dataframe with all relevant information for the rest of the pipeline
        self.user_data = merge_users_with_labels(self.labels_dt, self.user_data)
        for i, profile in self.user_data.iterrows():
            print(profile)


    def __len__(self):
        """
        gives the amount of user profiles in the dataset
        :return: the length of the dataset
        """
        return len(self.user_data)

    def __getitem__(self, idx : int):
        """
        reads out one user profile from the dataset
        :param idx: the index for one sample of the dataset (user profile)
        """
        profile = self.user_data.iloc[idx]
        return profile.to_dict()
    
if __name__ == "__main__":
        Midterm18("datasets")
    