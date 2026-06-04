import tarfile
import json
import pandas as pd
from warnings import warn
from torch.utils.data import Dataset
import os
from constants import CLEAN_USER_DICT, USER_COLUMNS, merge_users_with_labels

class Cresci19(Dataset):
    """
        Dataset for the Cresci-2019. The zipped dataset file should be named 'cresci-rtbust-2019.tar.gz' and should be placed in the directory /datasets.
    """

    def __init__(self):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
        """
        root = "datasets"
        #validity checks for the expected dataset file
        default_path = os.path.join(root, "cresci-rtbust-2019.tar.gz")
        extract_dir = os.path.join(root, "Cresci19")

        if not os.path.exists(default_path) and not os.path.exists(extract_dir):
            warn("Dataset file not found. Please make sure 'cresci-rtbust-2019.tar.gz' is in the dataset directory.")
            return 
        
        #checks if the dataset is not already extracted, extracts it if necessary and deletes the zip file after extraction
        if not os.path.exists(os.path.join(extract_dir)):
            os.makedirs(os.path.join(extract_dir), exist_ok=True)
            with tarfile.open(default_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            os.remove(default_path) 
            print("Finished 'Cresci19' extraction and removed the zip file")
       
           
        #validates the expected files are present in the extract_dir
        user_data_path = os.path.join(extract_dir, "cresci-rtbust-2019_tweets.json")
        labels_path = os.path.join(extract_dir, "cresci-rtbust-2019.tsv")

        if not os.path.exists(user_data_path):
            warn(
                "Dataset 'Cresci19.cresci-rtbust-2019_processed_user_objects.json' file not found. Please make sure 'cresci-rtbust-2019.tar.gz' is in the dataset directory and that it contains the expected files.",                    UserWarning
            )
            return
        
        if not os.path.exists(labels_path):
            warn(
                "Dataset 'Cresci19.cresci-rtbust-2019.tsv' file not found. Please make sure 'cresci-rtbust-2019.tar.gz' is in the dataset directory and that it contains the expected files.",                    UserWarning
            )
            return

        self.labels_dt = pd.read_csv(labels_path, sep="\t", header=None, names=["id", "label"])
        with open(user_data_path, "r") as f:
            data = json.load(f)
        normalized_data = pd.json_normalize(data, sep=".")

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
        self.user_data = self.user_data.dropna(subset=USER_COLUMNS)
        self.user_data = self.user_data[self.user_data["lang"] == "en"]


        self.user_data = merge_users_with_labels(self.labels_dt, self.user_data)



    def __len__(self):
        """
        gives the amount of user profiles in the dataset
        :return: the length of the dataset
        """
        return len(self.user_data)


    def __getitem__(self, idx):
        """
        retrieves a single user sample from the dataset

        :param idx: integer index of the user in the dataset
        :return: dictionary containing all user profile features for that index
        """
        profile = self.user_data.iloc[idx]
        return profile.to_dict()
    
if __name__ == "__main__":
    Cresci19()