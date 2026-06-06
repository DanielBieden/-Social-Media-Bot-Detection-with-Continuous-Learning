import os
from warnings import warn
from zipfile import ZipFile
import csv
from constants import Sample, UserData, TweetData



class Cresci18:   
    """
    Dataset for the Cresci18. The downloaded *.csv.zip files need to be in the directory /datasets 
                              or in another dataset whose filepath needs to be given as param to the constructor.
    """
    def __init__(self, root : str | None = None):
        """
        :param root: OPTIONAL,root is the filepath of the dataset directory in which the dataset is stored.
                     Dataset directory is default directory.
        """
        if root is None:
            root = "datasets"
        
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
        print("hello")
        if not os.path.exists(self.tweets_data_path):
            warn(
                "Dataset Cresci18's tweets.csv' file not found. Please make sure 'tweets.csv' is in the Cresci18 directory.",                    UserWarning
                )
            return


    def __iter__(self):
        """
        :return : sample of the dataclass Sample{TweetData, UserData, label : int}
        """
        
        users = {}
        bot_label = 0
        with open(self.user_data_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")

            for row in reader:
                print(row)
                print(row.keys())
                user_id = row["id"]
                bot_label = row["bot"]
                users[user_id] = UserData.from_row(row)
               
            
        with open(self.tweets_data_path, newline = ",", encoding="utf-8") as f:
           reader = csv.DictReader(f)
           for row in reader:
            user = users[row["user_id"]]
        
            sample = Sample(
                tweet_data=TweetData.from_row(row),
                user_data=user,
                label= int(bot_label)
                )

            yield sample

if __name__ == "__main__":
    Cresci18()
        
    





        