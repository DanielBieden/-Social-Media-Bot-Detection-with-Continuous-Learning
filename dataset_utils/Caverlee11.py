import os
from warnings import warn
from zipfile import ZipFile
import csv
from constants import Sample, UserData, TweetData
from torch.utils.data import IterableDataset



class Caverlee11(IterableDataset):   
    """
    Dataset for the Cresci18. The downloaded *.csv.zip files can be in the directory /datasets
                              or in another dataset whose filepath needs to be given as param to the constructor.
    """
    def __init__(self, root : str | None = None):
        """
        :param root: OPTIONAL,root is the filepath of the dataset directory in which the dataset is stored.
                     Dataset directory is default directory.
        :param root(OPTIONAL): the filepath of the dataset directory in which the dataset is stored, if none is given "datasets"
        :param mode: Dataset split to use ("train", "dev", or "test").
        :param train_split: Fraction of users assigned to the training set (e.g. 0.8 = 80%).
        :param dev_split: Fraction of users assigned to the validation set (e.g. 0.1 = 10
        """
         #__init__ does the filehandling
        if root is None:
            root = "datasets"
        
        extract_dir = "Caverlee11"
        
        if not os.path.exists(os.path.join(root, extract_dir)):
            zip_path = os.path.join(root, "caverlee-2011.zip")
            if not os.path.exists(zip_path):
                warn(
                "Dataset Caverlee11's caverlee-2011.zip file not found. Please make sure 'caverlee-2011.zip' is in the dataset directory.",                    UserWarning
                )
                return
        

            with ZipFile(zip_path) as zipObj:
                zipObj.extractall(os.path.join(root,extract_dir))
            os.remove(zip_path) # removes the zip file after extraction to save space
            print("Finished 'caverlee-2011.zip' extraction and removed the zip file")

        #creates data_paths to files
        self.polluters_data_path = os.path.join(root, extract_dir,"social_honeypot_icwsm_2011","content_polluters.txt" )
        self.polluters_tweets_data_path = os.path.join(root, extract_dir, "social_honeypot_icwsm_2011", "content_polluters_tweets.txt" )

        self.users_data_path = os.path.join(root, extract_dir,"social_honeypot_icwsm_2011","legitimate_users.txt" )
        self.users_tweets_data_path = os.path.join(root, extract_dir,"social_honeypot_icwsm_2011","legitimate_users_tweets.txt" )

        #checks that all files are where they should be
        if not os.path.exists(self.users_data_path):
            warn(
                "Dataset Caverlee11's 'legitimate_users' file not found. Please make sure 'legitimate_users.txt' is in the Caverlee11 directory.",                    UserWarning
                )
            return
        
        if not os.path.exists(self.polluters_data_path):
            warn(
                "Dataset Caverlee11's 'content_polluters' file not found. Please make sure 'content_polluters.txt' is in the Caverlee11 directory.",                    UserWarning
                )
            return

        if not os.path.exists(self.users_tweets_data_path):
            warn(
                "Dataset Caverlee11's legitimate_users_tweets file not found. Please make sure 'legitimate_users_tweets.txt' is in the Cresci18 directory.",                    UserWarning
                )
            return
        
        if not os.path.exists(self.polluters_tweets_data_path):
            warn(
                "Dataset Caverlee11's content_polluters_tweets file not found. Please make sure 'content_polluters_tweets.txt' is in the Cresci18 directory.",                    UserWarning
                )
            return


    def __iter__(self):
        """
        :return : sample of the dataclass Sample{TweetData, UserData, label}

        Reads the users metadata and the tweetdata. Then joins on user_id. Label is type given as a parameter after validation.

        """
        # makes it easier to handle with the labels
        datasets = [
        (
            self.polluters_data_path,
            self.polluters_tweets_data_path,
            "bot",
        ),
        (
            self.users_data_path,
            self.users_tweets_data_path,
            "human",
        ),
            ]
        # the .txt files do not have a header so, I am creating them from the READ_ME
        fieldnames_metadata = [
        "id",
        "created_at",
        "collected_at",
        "following_count",
        "followers_count",
        "tweets_count",
        "screen_name_len",
        "description_len",
                    ]
        
        fieldnames_tweets = [
        "user_id",
        "id",
        "text",
        "Created_at",
                    ]


        users = {}
        for metadata_path, tweets_data_path, bot_label in datasets:

            with open(metadata_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t", fieldnames=fieldnames_metadata)
                for row in reader:
                    user_id = row["id"]
                    users[user_id] = UserData.from_row(row)

            with open(tweets_data_path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t", fieldnames=fieldnames_tweets)
                for row in reader:
                    user = users[row["user_id"]]
                    if row is None:
                        continue
                
                    yield Sample(
                    tweet_data=TweetData.from_row(row),
                    user_data=user,
                    label=bot_label,
                )

            
        