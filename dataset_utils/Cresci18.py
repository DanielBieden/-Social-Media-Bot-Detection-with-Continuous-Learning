import os
from warnings import warn
from zipfile import ZipFile
import csv
from constants import Sample, UserData, TweetData
from torch.utils.data import IterableDataset
from splitting import hash_split_multi
from collections import defaultdict
from langdetect import detect,LangDetectException



class Cresci18(IterableDataset):   
    """
    Dataset for the Cresci18. The downloaded *.csv.zip files can be in the directory /datasets 
                              or in another dataset whose filepath needs to be given as param to the constructor.
    """
    def __init__(self, mode :str, train_split: float = 0.8, dev_split: float = 0.1, root : str |None = None):
        """
        :param root(OPTIONAL): the filepath of the dataset directory in which the dataset is stored, if none is given "datasets"
        :param mode: Dataset split to use ("train", "dev", or "test").
        :param train_split: Fraction of users assigned to the training set (e.g. 0.8 = 80%).
        :param dev_split: Fraction of users assigned to the validation set (e.g. 0.1 = 10

        """
         #__init__ does the filehandling
        if root is None:
            root = "datasets"
        
        self.mode = mode

        assert train_split + dev_split < 1.0

        extract_dir = "Cresci18"
        
        if not os.path.exists(os.path.join(root, extract_dir)):
            user_zip_path = os.path.join(root, "users.csv.zip")
            tweets_zip_path = os.path.join(root, "tweets.csv.zip")
            if not os.path.exists(user_zip_path):
                warn(
                "Dataset Cresci18's user.csv.zip file not found. Please make sure 'user.csv.zip' is in the dataset directory.",                    UserWarning
                )
                return
            
            if not os.path.exists(tweets_zip_path):
                warn(
                "Dataset Cresci18's tweets.csv.zip' file not found. Please make sure 'tweets.csv.zip' is in the dataset directory.",                    UserWarning
                )
                return

            with ZipFile(tweets_zip_path) as zipObj:
                zipObj.extractall(os.path.join(root,extract_dir))
            os.remove(tweets_zip_path) # removes the zip file after extraction to save space
            print("Finished 'tweets.csv.zip' extraction and removed the zip file")

            with ZipFile(user_zip_path) as zipObj:
                zipObj.extractall(os.path.join(root, extract_dir))
            os.remove(user_zip_path) # removes the zip file after extraction to save space
            print("Finished 'users.csv.zip' extraction and removed the zip file")
        
        self.user_data_path = os.path.join(root, extract_dir, "users.csv" )
        self.tweets_data_path = os.path.join(root, extract_dir, "tweets.csv" )


        if not os.path.exists(self.user_data_path):
            warn(
                "Dataset Cresci18's users.csv file not found. Please make sure 'users.csv' is in the Cresci18 directory.",                    UserWarning
                )
            return
        if not os.path.exists(self.tweets_data_path):
            warn(
                "Dataset Cresci18's tweets.csv' file not found. Please make sure 'tweets.csv' is in the Cresci18 directory.",                    UserWarning
                )
            return


    def __iter__(self):
        """
            :return : sample of the dataclass Sample{TweetData, UserData, label : "human" or "bot" or if unlabeled then ""}

            Reads the users metadata and the tweetdata. Then joins on user_id. Label is read from the user_data.
        """

        users = {}
        with open(self.user_data_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")
            for row in reader:
                users[row["id"]] = UserData.from_row(row)

        last_user = None
        current_tweets = []
        current_user_obj = None
        current_label = ""

        with open(self.tweets_data_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")

            for i, row in enumerate(reader):

                user_id = row["user_id"]

                # User-Wechsel -> flush
                if last_user is not None and user_id != last_user:

                    if current_tweets:
                        if current_user_obj is None:
                            continue

                        yield Sample(
                            tweet_data=current_tweets,
                            user_data=current_user_obj,
                            label=str(current_label),
                        )

                    current_tweets = []

                last_user = user_id

                text = row["text"]

                try:
                    if text and text.strip():
                        lang = detect(text)
                    else:
                        lang = None
                except LangDetectException:
                    lang = None

                if lang != "en":
                    continue

                split = hash_split_multi(user_id)
                if split != self.mode:
                    continue

                user = users.get(user_id)
                if user is None:
                    continue

                current_user_obj = user

                # label sauber pro user bestimmen
                raw_label = getattr(user, "bot", "")
                if raw_label == 0:
                    current_label = "human"
                elif raw_label == 1:
                    current_label = "bot"
                else:
                    current_label = ""

                current_tweets.append(
                    TweetData.from_row(row)
                )

                # optional: batch limit
                if len(current_tweets) == 200:
                    yield Sample(
                        tweet_data=current_tweets,
                        user_data=current_user_obj,
                        label=str(current_label),
                    )
                    current_tweets = []

            # letzter User flush
            if current_tweets and current_user_obj is not None:
                yield Sample(
                    tweet_data=current_tweets,
                    user_data=current_user_obj,
                    label=str(current_label),
                )    
        
if __name__ == "__main__":
    example = Cresci18("train",0.8,0.1)
    users = set()
    for i,sample in enumerate(example):
        print(len(sample.tweet_data))  
    
        if i == 2:
            break

       
        
    





        