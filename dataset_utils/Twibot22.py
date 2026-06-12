import shutil
import csv
import json
import ijson
import sys
import os
from warnings import warn
import sqlite3

try:
    from constants import UserData, TweetData, Sample, normalize_user_id, json_decimal_handler
except ImportError:
    from constants import UserData, TweetData, Sample, normalize_user_id, json_decimal_handler

from torch.utils.data import IterableDataset
from splitting import hash_split_multi

class Twibot22(IterableDataset):
    """
    Dataset for the Twibot-22. The downloaded *.json files or a directory,named "Twibot22", containing them, need to be in the directory /datasets.
    IMPORTANT: IF YOU CAN DOWNLOAD THEM DIRECTLY INTO THE TWIBOT22 FOLDER.
    """
    def __init__(self, mode :str, train_split: float = 0.8, dev_split: float = 0.1, root : str |None = None):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
        :param mode: Dataset split to use ("train", "dev", or "test").
        :param train_split: Fraction of users assigned to the training set (e.g. 0.8 = 80%).
        :param dev_split: Fraction of users assigned to the validation set (e.g. 0.1 = 10
        """
        if ijson.backend_name == 'python':
            print("\n" + "="*70)
            print("⚠️  WARNING: ijson is running in ultra-slow pure-Python mode!")
            print("Parsing these multi-gigabyte files will take ages.")
            print("\nHow to fix this:")
            if sys.platform.startswith('win'):
                print(" -> Open Anaconda Prompt and run: conda install -c conda-forge yajl")
            elif sys.platform.startswith('linux'):
                print(" -> Run: sudo apt-get install libyajl2")
            elif sys.platform.startswith('darwin'): # Mac
                print(" -> Run: brew install yajl")
            print("="*70 + "\n")

        #__init_ does the file handling
        if root == None:
            root = "datasets"

        self.mode = mode

        assert train_split + dev_split < 1.0

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
        self.user_data_path = os.path.join(root, "Twibot22", "user.json")
        self.labels_path = os.path.join(root, "Twibot22", "label.csv")
        self.tweet_paths = [
            os.path.join(root, "Twibot22", f"tweet_{i}.json")
            for i in range(9)
        ]


        #validation of the expected paths 
        if not os.path.exists(self.user_data_path):
            warn(
                "Dataset 'Twibo22.user.json' file not found. Please make sure 'user.json' is in the dataset directory.",                    UserWarning
            )
            return

        if not os.path.exists(self.labels_path):
            warn(
                "Dataset 'Twibot22.label.tsv' file not found. Please make sure 'label.csv' is in the dataset directory.",                    UserWarning
            )
            return
        
        for path in self.tweet_paths:
            if not os.path.exists(path):
                warn(
                    f"Dataset 'Twibot20.{path}' file not found. Please make sure every file is donwloaded in the dataset directory." 
                )


    
    

        