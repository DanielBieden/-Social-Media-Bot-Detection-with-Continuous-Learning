import shutil
import csv
import ijson
import sys
import os
from warnings import warn

from YAJLStreamSanitizer import YAJLStreamSanitizer
try:
    from constants import UserData, TweetData, Sample, normalize_user_id
except ImportError:
    from constants import UserData, TweetData, Sample, normalize_user_id

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

    def __iter__(self):

        labels = {}
        print("--> Loading labels...")
        with open(self.labels_path, newline="", encoding="utf-8") as f:
            label_data = csv.DictReader(f, delimiter=",")
            for item in label_data:
                user_id = normalize_user_id(item["id"])
                if hash_split_multi(user_id) != self.mode:
                    continue
                labels[normalize_user_id(item["id"])] = item["label"]
        print(f"--> Loaded {len(labels)} labels for split '{self.mode}'")
        print("--> Loading users (This will take a while via ijson)...")
        users = {}
        user_count = 0
        with open(self.user_data_path, "rb") as f:
            for row in ijson.items(f, "item"):
                user_id = normalize_user_id(row["id"])
                if hash_split_multi(user_id) != self.mode:
                    continue
                users[user_id] = row
                user_count += 1
                if user_count % 50000 == 0:
                    print(f"    Processed {user_count} users...")
                print(f"--> Successfully cached {len(users)} users in memory.")

        for path in self.tweet_paths:
            if not os.path.exists(path):
                continue
            print(f"--> Opening tweet file: {os.path.basename(path)}")
            with open(path, "rb") as f:
                sanitized_file = YAJLStreamSanitizer(f)

                current_user_id = None
                current_tweets = []
                
                try:

                    for row in ijson.items(sanitized_file, "item"):

                        normalized_row = dict(row)
                        user_id = normalize_user_id(normalized_row.get("author_id",0))

                        if user_id not in users or user_id not in labels:
                            continue

                        # First User
                        if current_user_id is None:
                            current_user_id = user_id

                        # New User gets detected, yields the old User and updates information
                        if user_id != current_user_id:

                            yield Sample(
                                tweet_data=current_tweets,
                                user_data=UserData.from_row(users[current_user_id]),
                                label=str(labels[current_user_id]),
                            )

                            current_user_id = user_id
                            current_tweets = []

                        #appends the tweet since ids match
                        current_tweets.append(
                            TweetData.from_row(normalized_row)
                        )
                    
                except ijson.IncompleteJSONError as e:
                    # If a file cuts off abruptly (truncated download), we log it and move to the next file
                    print(f"⚠️ Note: Stream ended early or hit a structural break in {os.path.basename(path)}.")
                    print(f"   Successfully extracted data up to the break point.")
                
                if current_tweets:
                        yield Sample(
                            tweet_data=current_tweets,
                            user_data=UserData.from_row(users[current_user_id]),
                            label=str(labels[current_user_id]),
                        )

                        
if __name__ == "__main__":
    example = Twibot22("train",0.8,0.1)
    users = set()
    size = 0
    for i,sample in enumerate(example):
       size += 1
       if sample.user_data.id in users:
           print("Double")
       for tweet in sample.tweet_data:
           users.add(tweet.user_id)

       if i == 100000:
               break
        
       
    print(size)
    print(len(users))
            

    
    

        