import hashlib

def hash_split(user_id: str, train_ratio=0.8):
    """
    :param user_id: user_id str is the id of the user 
    :param train_ratio: is the procentage in decimals of which should go into train
    """
    h = hashlib.md5(user_id.encode()).digest()
    v = int.from_bytes(h[:4], "little") / 2**32
    return v < train_ratio


def hash_split_multi(user_id: str, train=0.8, dev=0.1):
    """
    :param user_id: user_id str is the id of the user 
    :param train_ratio: is the procentage in decimals of which should go into train
    :param dev_ratio: is the procentage in decimals of which should go into dev
    """
    if user_id is None:
        return "train"
    
    user_id = str(user_id)

    h = hashlib.md5(user_id.encode()).digest()
    v = int.from_bytes(h[:4], "little") / 2**32

    if v < train:
        return "train"
    elif v < train + dev:
        return "dev"
    else:
        return "test"