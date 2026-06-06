import numpy as np
from dataclasses import dataclass

@dataclass
class UserData:
    id: int
    name: str
    screen_name: str
    statuses_count: int
    followers_count: int
    friends_count: int
    favourites_count: int
    listed_count: str
    lang: str
    verified: int

    @staticmethod
    def from_row(row):
        return UserData(
            id=row["id"],
            name=row["name"],
            screen_name=row["screen_name"],
            statuses_count=int(row["statuses_count"] or 0),
            followers_count=int(row["followers_count"] or 0),
            friends_count=int(row["friends_count"] or 0),
            favourites_count=int(row["favourites_count"] or 0),
            listed_count=str(row["listed_count"] or 0),
            lang=str(row["lang"] or None),
            verified=int(row["verified"] or 0),
        )


@dataclass
class TweetData:
    id: int
    text: str
    user_id: int
    in_reply_to_status_id: int
    in_reply_to_user_id: int
    in_reply_to_screen_name: int
    retweet_count: int
    reply_count: int
    favorite_count: int
    num_hashtags: int
    num_urls: int
    num_mentions: int
    timestamp: int

    @staticmethod
    def from_row(row):
        return TweetData(
            id=row["id"],
            text=row["text"],
            user_id=int(row["user_id"] or 0),
            in_reply_to_status_id=int(row["in_reply_to_status_id"] or 0),
            in_reply_to_user_id=int(row["in_reply_to_user_id"] or 0),
            in_reply_to_screen_name=int(row["in_reply_to_screen_name"] or 0),
            retweet_count=int(row["retweet_count"] or 0),
            reply_count=int(row["reply_count"] or 0),
            favorite_count=int(row["favorite_count"] or 0),
            num_hashtags=int(row["num_hashtags"] or 0),
            num_urls=int(row["num_urls"] or 0),
            num_mentions=int(row["num_mentions"] or 0),
            timestamp=int(row["timestamp"] or 0),
        )


@dataclass
class Sample:
    tweet_data: TweetData
    user_data: UserData
    label: int

USER_COLUMNS = [
        'id',
        'name',
        'screen_name',
        'statuses_count',
        'followers_count',
        'friends_count',
        'favourites_count',
        'listed_count',
        'lang',
        'verified'
        ]
CLEAN_USER_DICT = {
            'verified':{np.nan: 0},
            'protected':{np.nan: 0},
        } 
TWEET_COLUMNS = [
    'id',
    'text',
    'user_id',
    'in_reply_to_status_id',
    'in_reply_to_user_id',
    'in_reply_to_screen_name',
    'retweet_count',
    'reply_count',
    'favorite_count',
    'num_hashtags',
    'num_urls',
    'num_mentions',
    'timestamp'
]
CLEAN_TWEET_DICT = {
    'favourites_count':{np.nan: 0}
        }

def merge_users_with_labels(user_data, labels_dt):
    """
    merges the user profile data with the labels for the dataset. This is necessary to have a single dataframe with all relevant information for the rest of the pipeline.
    :param user_data: the dataframe containing the user profile data
    :param labels_dt: the dataframe containing the labels for the dataset
    :return: a merged dataframe containing both user profile data and labels
    """
    # merge into single self.user_data dataframe with only relevant columns
    user_data["id"] = user_data["id"].astype(str)
    labels_dt["id"] = labels_dt["id"].astype(str)

    return user_data.merge(
        labels_dt, on="id", how="inner")