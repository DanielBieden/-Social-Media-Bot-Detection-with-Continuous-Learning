import os
from warnings import warn
from zipfile import ZipFile
import csv
from constants import Sample, UserData, TweetData
from torch.utils.data import IterableDataset
from splitting import hash_split_multi
from collections import defaultdict
import duckdb
import json

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
        output_file = os.path.join("Cresci18", "english_tweet_ids.json")

        with open(output_file, "w", encoding="utf-8") as out:
            with open(self.user_data_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=",")
                bot_label = ""
                for row in reader:
                    if row["lang"] != "en" or row["lang"] != "Null":
                        continue

                    user_id = row["id"]
                    users[user_id] = row
                    if row["bot"] == 0:
                        bot_label = "human"
                    elif row["bot"] == 1:
                        bot_label = "bot"
                    else:
                        bot_label = ""
                    
            tweets_per_user = defaultdict(list) 

            con = duckdb.connect()
            query = f"""
            SELECT
            t.*
            FROM read_csv_auto('{self.tweets_data_path}') t
            ORDER BY t.user_id
            """     

            rel = con.execute(query)
            cols = [c[0] for c in rel.description]

            for row in rel.fetchall():
                row = dict(zip(cols, row))
                user_id = row["user_id"]


                text = row.get("text")
                if not text or not text.strip():
                    continue

                user = users.get(user_id)
                if user is None:
                    continue

                tweets_per_user[user_id].append(TweetData.from_row(row))

                json.dump(row["id"], out)
                f.write("\n")

            
                yield Sample(
                        tweet_data=tweets_per_user[user_id],
                        user_data=user,
                        label=str(bot_label),
                    )
                
if __name__ == "__main__":
    example = Cresci18("train",0.8,0.1)
    users = set()
    size = 0
    for i,sample in enumerate(example):
       size += 1
       print(len(sample.tweet_data))
       for tweet in sample.tweet_data:
           users.add(tweet.user_id)
        
       
    print(size)
    print(len(users))

       
        
    





        