from importlib.resources import path
   

import ijson
from zipfile import ZipFile
from warnings import warn
from torch.utils.data import IterableDataset
try:
    from .constants import UserData, TweetData, Sample, normalize_user_id
except ImportError:
    from constants import UserData, TweetData, Sample, normalize_user_id

import os

class Twibot20(IterableDataset):
    """
    Dataset for the Twibot-20. The downloaded *.json files or a directory containing them,named "Twibot20", need to be in the directory /datasets.
    """
    def __init__(self, root : str | None = None):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
        """
        if root is None:
            root = "datasets"

        if not os.path.exists(os.path.join(root, "Twibot20")):
            default_path = os.path.join(root, "data.zip")
            if not os.path.exists(default_path):
                warn(
                "Dataset 'Twibot20' file not found. Please make sure 'Twibot-20.zip' is in the dataset directory.",                    UserWarning
                )
                return 

            with ZipFile(default_path) as zipObj:
                zipObj.extractall(os.path.join(root))
            os.remove(default_path) # removes the zip file after extraction to save space
            print("Finished 'Twibot-20' extraction and removed the zip file")

        self.data_path = os.path.join(root, "Twibot20", "node.json")
        if not os.path.exists(self.data_path):
            warn(
                "Dataset Twibot20's 'support.json' file not found. Please make sure 'support.json' is in the dataset directory.",                    UserWarning
            )
            return


    def __iter__(self):
        users = {}
        labels = {}


        with open(self.data_path, "rb") as f:
            for row in ijson.items(f, "item"):
                user_id = normalize_user_id(row["id"])
                users[user_id] = UserData.from_row(row)

        for path in self.tweet_paths:
            if not os.path.exists(path):
                continue

            with open(path, "rb") as f:
                for row in ijson.items(f, "item"):

                    normalized_row = dict(row)
                    normalized_row["user_id"] = normalized_row.get("user_id", normalized_row.get("author_id", 0))

                    user_id = normalize_user_id(normalized_row.get("user_id", 0))
                    user = users.get(user_id)
                    bot_label = labels.get(user_id)

                    if user is None or bot_label is None:
                        continue

                    yield Sample(
                        tweet_data=TweetData.from_row(normalized_row),
                        user_data=user,
                        label=str(bot_label),
                    )


        
        

if __name__ == "__main__":
    Twibot20()

        
       




       



        