import os
from warnings import warn
from zipfile import ZipFile
import pandas as pd
from constants import USER_COLUMNS, CLEAN_USER_DICT, TWEET_COLUMNS, CLEAN_TWEET_DICT



class Cresci18:   
    """
    Dataset for the Twibot-20. The downloaded *.json files or a directory containing them,named "Twibot20", need to be in the directory /datasets.
    """
    def __init__(self, root : str | None = None):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
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
        
        user_data_path = os.path.join(root, extract_dir, "users.csv" )
        tweets_data_path = os.path.join(root, extract_dir, "tweets.csv" )


        if not os.path.exists(user_data_path):
            warn(
                "Dataset Cresci18's users.csv file not found. Please make sure 'users.csv' is in the Cresci18 directory.",                    UserWarning
                )
            return
            
        if not os.path.exists(tweets_data_path):
            warn(
                "Dataset Cresci18's tweets.csv' file not found. Please make sure 'tweets.csv' is in the Cresci18 directory.",                    UserWarning
                )
            return
        

        self.user_data = pd.read_csv(
        user_data_path,
        sep=",",
        quotechar='"',
        engine="python",
        on_bad_lines="skip")   

        

        self.user_data = self.user_data.replace(CLEAN_USER_DICT)
        self.user_data["protected"] = None
        self.user_data = self.user_data[USER_COLUMNS]

        self.reader = pd.read_csv(tweets_data_path, engine="python", on_bad_lines="skip")
        self.reader = self.reader[TWEET_COLUMNS]
        self.reader = self.reader.replace(CLEAN_TWEET_DICT)

        print(len(self.user_data))
        print(len(self.reader))
        

    def __len__(self):
        return len(self.user_data)


    def __getitem__(self, idx : int):
        profile = self.user_data.iloc[idx]
        user_id = profile["id"]

        # get all tweets from the selected user profile
        tweets = self.reader[self.reader["user_id"] == user_id]
        result = {
            "profile": profile.to_dict(),
            "tweets": tweets.to_dict("records"),
        }
        return result

if __name__ == "__main__":
    Cresci18()
        
    





        