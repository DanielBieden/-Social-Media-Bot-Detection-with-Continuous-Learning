import os
from warnings import warn
from zipfile import ZipFile
import csv
from constants import Sample, UserData, TweetData
from torch.utils.data import IterableDataset
from splitting import hash_split_multi
from langdetect import detect,LangDetectException



class Caverlee11(IterableDataset):   
    """
    Dataset for the Cresci18. The downloaded *.csv.zip files can be in the directory /datasets
                              or in another dataset whose filepath needs to be given as param to the constructor.
    """
    def __init__(self, mode :str, train_split: float = 0.8, dev_split: float = 0.1, root : str |None = None):
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
        
        self.mode = mode

        assert train_split + dev_split < 1.0

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

        datasets = [
            (self.polluters_data_path, self.polluters_tweets_data_path, "bot"),
            (self.users_data_path, self.users_tweets_data_path, "human"),
        ]

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

        for metadata_path, tweets_path, label in datasets:

            # --- metadata laden ---
            users = {}
            with open(metadata_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t", fieldnames=fieldnames_metadata)
                for row in reader:
                    users[row["id"]] = UserData.from_row(row)

            # --- tweets streaming ---
            with open(tweets_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t", fieldnames=fieldnames_tweets)

                current_user_id = None
                current_tweets = []
                current_user = None

                for row in reader:

                    user_id = row["user_id"]

                    # split filter
                    if hash_split_multi(user_id) != self.mode:
                        continue

                    if not row or not row.get("text"):
                        continue

                    try:
                        lang = detect(row["text"])
                        if lang != "en":
                            continue
                    except LangDetectException:
                        continue

                    # userwechsel → flush
                    if current_user_id is not None and user_id != current_user_id:
                        if current_tweets:
                            yield Sample(
                                tweet_data=current_tweets,
                                user_data=users[current_user_id],
                                label=label,
                            )
                        current_tweets = []

                    # neuer user initialisieren
                    if user_id != current_user_id:
                        current_user_id = user_id
                        current_user = users[user_id]

                    try:
                        current_tweets.append(TweetData.from_row(row))
                    except ValueError:
                        continue

                    if current_user is None:
                        continue
                    

                    yield Sample(
                            tweet_data=current_tweets,
                            user_data=current_user,
                            label=label,
                        )
                    current_tweets = []

                # letzter user flush
                if current_user_id is not None and current_tweets:
                    yield Sample(
                        tweet_data=current_tweets,
                        user_data=users[current_user_id],
                        label=label,
                    )
                    

if __name__ == "__main__":
    example = Caverlee11("train",0.8,0.1)
    users = set()
    for i,sample in enumerate(example):
        users.add(sample.user_data.id)
        print(len(sample.tweet_data))
    print(len(users))
        

            
        