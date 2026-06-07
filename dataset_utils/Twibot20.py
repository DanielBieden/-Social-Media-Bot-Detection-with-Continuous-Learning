import json
import ijson
from zipfile import ZipFile
from warnings import warn
from torch.utils.data import IterableDataset
try:
    from .constants import UserData, TweetData, Sample, normalize_user_id,_safe_dict, clean_text
except ImportError:
    from constants import UserData, TweetData, Sample, normalize_user_id, _safe_dict, clean_text
import os

from dataset_utils.splitting import hash_split_multi
class Twibot20(IterableDataset):
    """
    Dataset for the Twibot-20. The downloaded *.json files or a directory containing them,named "Twibot20", need to be in the directory /datasets.
    """
    def __init__(self, mode :str, train_split: float = 0.8, dev_split: float = 0.1, root : str |None = None):
        """
        :param root(OPTIONAL): the filepath of the dataset directory in which the dataset is stored, if none is given "datasets"
        :param mode: Dataset split to use ("train", "dev", or "test").
        :param train_split: Fraction of users assigned to the training set (e.g. 0.8 = 80%).
        :param dev_split: Fraction of users assigned to the validation set (e.g. 0.1 = 10)

        """
        if root is None:
            root = "datasets"

        self.mode = mode

        assert train_split + dev_split < 1.0

        if not os.path.exists(os.path.join(root, "Twibot-20")):
            default_path = os.path.join(root, "data.zip")
            if not os.path.exists(default_path):
                warn(
                "Dataset 'Twibot20' file not found. Please make sure 'Twibot-20.zip' is in the dataset directory.",                    UserWarning
                )
                return 

            with ZipFile(default_path) as zipObj:
                zipObj.extractall(os.path.join(root))
            os.remove(default_path)  # removes the zip file after extraction to save space
            print("Finished 'Twibot-20' extraction and removed the zip file")

        self.data_path_train = os.path.join(root, "Twibot-20", "train.json")
        self.data_path_dev = os.path.join(root, "Twibot-20", "dev.json")
        self.data_path_test = os.path.join(root,"Twibot-20", "test.json")

        if not os.path.exists(self.data_path_test):
            warn(
                "Dataset Twibot20's 'test.json' file not found. Please make sure 'test.json' is in the dataset directory.",
                UserWarning,
            )

        if not os.path.exists(self.data_path_train):
            warn(
                "Dataset Twibot20's 'train.json' file not found. Please make sure 'train.json' is in the dataset directory.",
                UserWarning,
            )
            return

        if not os.path.exists(self.data_path_dev):
            warn(
                "Dataset Twibot20's 'dev.json' file not found. Please make sure 'dev.json' is in the dataset directory.",
                UserWarning,
            )
            return
        
        #helps looking up all of them and in order
        self.files = [
        self.data_path_test,
        self.data_path_train,
        self.data_path_dev,
    ]
        

    def __iter__(self):
        for path in self.files:
            with open(path, "r", encoding="utf-8") as f:

                data = json.load(f)  # Twibot-20 files are NOT huge JSON arrays per line, usually list-based
                tweet_id = 0
                for row in data:
            

                    user_id = normalize_user_id(row.get("ID"))
                    profile = _safe_dict(row.get("profile"))
                    split = hash_split_multi(user_id) 
                    if split != self.mode:
                        continue

                    user = UserData.from_row({
                    "id": user_id,
                    "profile": profile
                    })

                    bot_label = row.get("label")
                    if bot_label == "1":
                        bot_label = "bot"
                    elif bot_label == "0":
                        bot_label = "human"
                    elif bot_label == "":
                        bot_label = ""
                    elif bot_label is None:
                        bot_label = ""
                        continue
                    
                    
                    tweets = row.get("tweet",[])
                    if tweets is None:
                        continue
                    for tweet in tweets:
                        tweet_id += 1
                        tweet = str(tweet).strip()
                        if not tweet:
                            continue
                        #adds missinh fields
                        tweet_row = {
                        "id": {tweet_id},
                        "text": tweet,
                        "user_id": user_id,
                        "created_at": None,
                        "public_metrics": {},
                        "entities": {},
                    }   
                        

                        yield Sample(
                            tweet_data=TweetData.from_row(tweet_row),
                            user_data=user,
                            label=str(bot_label),
                    )

    
    
  

        
       




       



        