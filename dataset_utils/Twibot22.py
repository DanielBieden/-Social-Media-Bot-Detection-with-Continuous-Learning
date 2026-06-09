import shutil
import csv
import ijson
import os
from warnings import warn

try:
    from .constants import UserData, TweetData, Sample, normalize_user_id
except ImportError:
    from constants import UserData, TweetData, Sample, normalize_user_id

from torch.utils.data import IterableDataset
from splitting import hash_split_multi
from collections import defaultdict
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

    def __iter__(self):
        labels = {}
        with open(self.labels_path, newline="", encoding="utf-8") as f:
            label_data = csv.DictReader(f, delimiter=",")
            for item in label_data:
                labels[normalize_user_id(item["id"])] = item["label"]

        users = {}
        with open(self.user_data_path, "rb") as f:
            for row in ijson.items(f, "item"):
                user_id = normalize_user_id(row["id"])
                users[user_id] = UserData.from_row(row)

        for path in self.tweet_paths:
            if not os.path.exists(path):
                continue

            with open(path, "rb") as f:

                current_user_id = None
                current_tweets = []

                for row in ijson.items(f, "item"):

                    normalized_row = dict(row)
                    normalized_row["user_id"] = normalized_row.get(
                        "user_id",
                        normalized_row.get("author_id", 0)
                    )

                    user_id = normalize_user_id(
                        normalized_row.get("user_id", 0)
                    )

                    split = hash_split_multi(user_id)
                    if split != self.mode:
                        continue

                    user = users.get(user_id)
                    bot_label = labels.get(user_id)

                    if user is None or bot_label is None:
                        continue

                    # erster User
                    if current_user_id is None:
                        current_user_id = user_id

                    # neuer User -> alten User yielden
                    if user_id != current_user_id:

                        yield Sample(
                            tweet_data=current_tweets,
                            user_data=users[current_user_id],
                            label=str(labels[current_user_id]),
                        )

                        current_user_id = user_id
                        current_tweets = []

                    current_tweets.append(
                        TweetData.from_row(normalized_row)
                    )

                # letzten User nach Dateiende yielden
                if current_tweets:
                    yield Sample(
                        tweet_data=current_tweets,
                        user_data=users[current_user_id],
                        label=str(labels[current_user_id]),
                    )
                        
if __name__ == "__main__":
    example = Twibot22("train",0.8,0.1)
    users = set()
    for i,sample in enumerate(example):
        print(len(sample.tweet_data))
        print(sample)   
    
        if i == 2:
            break                
    

    
    

        