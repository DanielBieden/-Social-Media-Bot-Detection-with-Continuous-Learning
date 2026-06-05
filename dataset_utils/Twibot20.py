from importlib.resources import path
from constants import USER_COLUMNS, CLEAN_USER_DICT   
import pandas as pd
import json
import ijson
from zipfile import ZipFile
from warnings import warn
from torch.utils.data import Dataset
import os

class Twibot20(Dataset):
    """
    Dataset for the Twibot-20. The downloaded *.json files or a directory containing them,named "Twibot20", need to be in the directory /datasets.
    """
    def __init__(self,sample_size: int | None = None, root : str | None = None):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
        """
        if root is None:
            root = "datasets"

        if not os.path.exists(os.path.join(root, "Twibot20")):
            default_path = os.path.join(root, "Twibot20.zip")
            if not os.path.exists(default_path):
                warn(
                "Dataset 'Twibot20' file not found. Please make sure 'Twibot-20.zip' is in the dataset directory.",                    UserWarning
                )
                return 

            with ZipFile(default_path) as zipObj:
                zipObj.extractall(os.path.join(root))
            os.remove(default_path) # removes the zip file after extraction to save space
            print("Finished 'Twibot-20' extraction and removed the zip file")

        user_data_path = os.path.join(root, "Twibot20", "support.json")
        if not os.path.exists(user_data_path):
            warn(
                "Dataset Twibot20's 'support.json' file not found. Please make sure 'support.json' is in the dataset directory.",                    UserWarning
            )
            return
        
        chunk_idx = 0
        samples_in_chunk = 0

        out = open(
        os.path.join(root, "Twibot20", f"chunk_{chunk_idx}.jsonl"),
            "w",
        encoding="utf-8")

        with open("Twibot20", "rb") as f:
            for sample in ijson.items(f, "item"):
                out.write(json.dumps(sample) + "\n")
                samples_in_chunk += 1

            if samples_in_chunk == sample_size:
                out.close()

                chunk_idx += 1
                samples_in_chunk = 0

        out.close()

        chunk_paths = []
        for i in range(chunk_idx):
            chunk_path = os.path.join(root,"Twibot20",f"chunk_{i}.jsonl")
            chunk_paths.append(chunk_path)

       

        data = pd.read_json(chunk_paths[0], lines = True)

        # flatten the nested user object into the columns expected by the pipeline
        rename_map = {
            f"profile.{column}": column
            for column in USER_COLUMNS
            if f"profile.{column}" in data.columns
        }
        self.user_data = data.rename(columns=rename_map).copy()
        available_columns = [column for column in USER_COLUMNS if column in self.user_data.columns]
        self.user_data = self.user_data[available_columns]
        self.user_data = self.user_data.replace(CLEAN_USER_DICT)
    
        self.user_data = self.user_data.dropna(subset=available_columns)
       
        size = 0
        for i,profile in self.user_data.iterrows():
            print(profile)
            size = size + 1
            if i == 9:
                break

        print(size)
        

if __name__ == "__main__":
    Twibot20(100)

        
       




       



        