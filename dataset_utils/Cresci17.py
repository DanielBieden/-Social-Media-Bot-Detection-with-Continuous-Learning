
from zipfile import ZipFile
from torch.utils.data import IterableDataset
import os
import csv
from constants import TweetData,UserData,Sample,Cresci17SetTypes

class Cresci17(IterableDataset):
    """
    Dataset for the Cresci-2017. The zipped dataset file should be named 'cresci-2017.csv.zip' and should be placed in the directory /datasets.
    """
    def __init__(self,subset_type ,  root : str |None = None):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
        :param subset_type: the wanted subset of the dataset.
        """
        if root == None:
            root = "datasets"

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
        """
        :return : sample of the dataclass Sample{TweetData, UserData, label : int}
        """
        
        users = {}
        with open(self.user_data_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")

            for row in reader:
                user_id = row["id"]
                users[user_id] = UserData.from_row(row)
               
            
        with open(self.tweet_data_path, newline = "", encoding="utf-8") as f:
           reader = csv.DictReader(f, delimiter=",")
           for row in reader:
            user = users[row["user_id"]]
        
            sample = Sample(
                tweet_data=TweetData.from_row(row),
                user_data=user,
                label=str(self.subset_type.value)
                )
            yield sample

if __name__ == "__main__":
    example = Cresci17(Cresci17SetTypes.GENUINE_USER)

    for i, sample in enumerate(example):
        print(f"\n--- SAMPLE {i} ---")
        print("tweet:", sample.tweet_data.text[:120])
        print("user_id:", sample.user_data.id)
        print("label:", sample.label)
        print("Sample", sample)
        if i == 2:
            break
    
