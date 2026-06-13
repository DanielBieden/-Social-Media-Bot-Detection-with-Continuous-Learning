
from zipfile import ZipFile
from torch.utils.data import IterableDataset
import os
import csv
from dataset_utils.constants import TweetData,UserData,Sample,Cresci17SetTypes
from dataset_utils.splitting import hash_split_multi
from collections import defaultdict
class Cresci17(IterableDataset):
    """
    Dataset for the Cresci-2017. The zipped dataset file should be named 'cresci-2017.csv.zip' and should be placed in the directory /datasets.
    """
    def __init__(self,subset_type , mode :str, train_split: float = 0.8, dev_split: float = 0.1, root : str |None = None):
        """
        :param root(OPTIONAL): the filepath of the dataset directory in which the dataset is stored, if none is given "datasets"
        :param mode: Dataset split to use ("train", "dev", or "test").
        :param train_split: Fraction of users assigned to the training set (e.g. 0.8 = 80%).
        :param dev_split: Fraction of users assigned to the validation set (e.g. 0.1 = 10

        Should you want all Subsets, you need to call the constructor for all Cresci17SetTypes. 
        """
        #__init__ does the filehandling
        if root == None:
            root = "datasets"

        self.mode = mode

        assert train_split + dev_split < 1.0

        self.subset_type = subset_type
        # select appropriate subdirectory to find wanted subset
        default_file_path = os.path.join(root, "cresci-2017" ,"datasets_full.csv")
        subset_path  = ""
        match subset_type:
            case Cresci17SetTypes.GENUINE_USER:
                subset_path = "genuine_accounts.csv"
            case Cresci17SetTypes.FAKE_FOLLOWER:
                subset_path = "fake_followers.csv"
            case Cresci17SetTypes.SOCIAL_SPAM_1:
                subset_path = "social_spambots_1.csv"
            case Cresci17SetTypes.SOCIAL_SPAM_2:
                subset_path = "social_spambots_2.csv"
            case Cresci17SetTypes.SOCIAL_SPAM_3:
                subset_path = "social_spambots_3.csv"
            case Cresci17SetTypes.TRADITIONAL_SPAM_1:
                subset_path = "traditional_spambots_1.csv"
            case Cresci17SetTypes.TRADITIONAL_SPAM_2:
                subset_path = "traditional_spambots_2.csv"
            case Cresci17SetTypes.TRADITIONAL_SPAM_3:
                subset_path = "traditional_spambots_3.csv"
            case Cresci17SetTypes.TRADITIONAL_SPAM_4:
                subset_path = "traditional_spambots_4.csv"
            case _:
                raise ValueError(f"Unknown subset type for Cresci17 dataset was used: {subset_type}")


        self.user_data_path = os.path.join(default_file_path, subset_path, "users.csv")
        self.tweet_data_path = os.path.join(default_file_path, subset_path, "tweets.csv")

        # automatic unzipping of files if they don't exist in an unpackaged state
        # check if files exist
        if not os.path.exists(self.user_data_path) or not os.path.exists(self.tweet_data_path):
            # check if dataset in whole exists, else unzip it
            if not os.path.exists(default_file_path):
                # check if zip file for the whole dataset exists
                if not os.path.exists(os.path.join(root, "cresci-2017.csv.zip")):
                    raise FileNotFoundError(f"no dataset structure or dataset zip file 'cresci-2017.csv.zip' found' at {root}")
                # unzip whole dataset
                with ZipFile(os.path.join(root, "cresci-2017.csv.zip")) as zipObj:
                    os.mkdir(os.path.join(root, "cresci-2017"))
                    zipObj.extractall(os.path.join(root, "cresci-2017"))
                    
            # unzip subset
            with ZipFile(os.path.join(default_file_path, subset_path + ".zip")) as zipObj:
                zipObj.extractall(default_file_path)
                zipObj.close()

    def __iter__(self):

        users = {}

        with open(self.user_data_path, newline="", encoding="latin-1") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if hash_split_multi(row["id"]) != self.mode:
                    continue
                if row["lang"] != "en" and row["lang"] != "NULL":
                    continue
                users[row["id"]] = row

        tweets = defaultdict(list)
        with open(self.tweet_data_path, newline="", encoding="latin-1") as f:
            reader = csv.DictReader(f,  delimiter=",")
            for row in reader:
                    
                user_id = row["user_id"]

                if user_id not in users:
                    continue
                
                if not row or not row.get("text"):
                    continue

                row["in_reply_to_screen_name"] = 0

                tweets[user_id].append(row)
                
       
        for user_id in users:  
            try:
                user_tweets = tweets.get(user_id, [])   
                processed_tweets = [TweetData.from_row(tweet_row) for tweet_row in user_tweets]
            except ValueError:
                continue
            yield Sample(
                    tweet_data=processed_tweets,
                    user_data=UserData.from_row(users[user_id]),
                    label=str(self.subset_type.value),
                    )
        
        
if __name__ == "__main__":
    example = Cresci17(Cresci17SetTypes.TRADITIONAL_SPAM_1,"train",0.8,0.1)
    users = set()
    size = 0
    for i,sample in enumerate(example):
        users.add(sample.user_data.id)
        size += 1
        for tweet in sample.tweet_data:
           users.add(tweet.user_id)
        if size != len(users):
            print(sample.user_data.id)
     
            
            
    print(size)
    print(len(users))


        
   
    



        



   

    
    
