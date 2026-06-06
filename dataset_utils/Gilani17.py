import tarfile
import json
import pandas as pd
from warnings import warn
from platformdirs import user_data_path
from torch.utils.data import Dataset
import os
from constants import CLEAN_USER_DICT, USER_COLUMNS, merge_users_with_labels

class Gilani17(Dataset):
    """
        Dataset for the Gilani-2017. The zipped dataset file should be named 'gilani-2017.tar.gz' and should be placed in the directory /datasets.
    """

    def __init__(self, root : str | None = None):
        
        if root == None:
            root = "datasets" 
            
        default_path = os.path.join(root, "gilani-2017.tar.gz")
        extract_dir = os.path.join(root, "Gilani17")

        if not os.path.exists(default_path) and not os.path.exists(extract_dir):
            warn("Dataset file not found. Please make sure 'gilani-2017.tar.gz' is in the dataset directory.")
            return

        # extract dataset into Gilani17 if it is not already extracted
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(default_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            os.remove(default_path)
            print("Finished 'Gilani17' extraction")


        #validates the expected files are present in the extract_dir
        user_data_path = os.path.join(os.path.join(root, "Gilani17"), "gilani-2017_tweets.json")
        labels_path = os.path.join(os.path.join(root, "Gilani17"), "gilani-2017.tsv")

        if not os.path.exists(user_data_path):
            warn(
                "Dataset 'Gilani17.gilani-2017_processed_user_objects.json' file not found. Please make sure 'gilani-2017.tar' is in the dataset directory and that it contains the expected files.",                    UserWarning
            )
            return
        
        if not os.path.exists(labels_path):
            warn(
                "Dataset 'Gilani17.gilani-2017.tsv' file not found. Please make sure 'gilani-2017.tar' is in the dataset directory and that it contains the expected files.",                    UserWarning
            )
            return

        #reads the data from the files
        self.labels_dt = pd.read_csv(labels_path, sep="\t", header=None, names=["id", "label"])

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
        self.user_data = self.user_data[self.user_data["lang"] == "en"]

        # merges users with labels to have a single dataframe with all relevant information for the rest of the pipeline
        self.user_data = merge_users_with_labels(self.user_data, self.labels_dt)



    def __len__(self):
        return len(self.user_data)


    def __getitem__(self, idx : int):
        """
        retrieves a single user sample from the dataset

        :param idx: integer index of the user in the dataset
        :return: dictionary containing all user profile features for that index
        """
        profile = self.user_data.iloc[idx]
        return profile.to_dict()
    