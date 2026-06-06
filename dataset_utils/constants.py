import numpy as np
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

def normalize_user_id(value):
    text = str(value or "").strip()
    if not text:
        return "0"
    return text if text.startswith("u") else f"u{text}"


def _count_items(value):
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


def _parse_timestamp(value):
    if value in (None, ""):
        return 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)

    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        try:
            return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
        except ValueError:
            return 0


@dataclass
class UserData:
    id: str
    name: str
    screen_name: str
    statuses_count: int
    followers_count: int
    friends_count: int
    favourites_count: int
    listed_count: int
    lang: str
    verified: int

    @staticmethod
    def from_row(row):

        return UserData(
    id=normalize_user_id(row.get("id", 0)),

    name=row.get("name", None),
    screen_name=row.get("screen_name", None),

    statuses_count=int(row.get("statuses_count", 0) or 0),
    followers_count=int(row.get("followers_count", 0) or 0),
    friends_count=int(row.get("friends_count", 0) or 0),
    favourites_count=int(row.get("favourites_count", 0) or 0),
    listed_count=int(row.get("listed_count", 0) or 0),

    lang=row.get("lang", None),

    verified=int(row.get("verified", 0) or 0),
)


@dataclass
class TweetData:
    id: str
    text: str
    user_id: str
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
        public_metrics = row.get("public_metrics") or {}
        entities = row.get("entities") or {}

        return TweetData(
            id=str(row.get("id", 0) or 0),
            text=str(row.get("text", "")),
            user_id=normalize_user_id(row.get("user_id", row.get("author_id", 0))),
            in_reply_to_status_id=int(row.get("in_reply_to_status_id", 0) or 0),
            in_reply_to_user_id=int(row.get("in_reply_to_user_id", 0) or 0),
            in_reply_to_screen_name=int(row.get("in_reply_to_screen_name", 0) or 0),
            retweet_count=int(public_metrics.get("retweet_count", row.get("retweet_count", 0)) or 0),
            reply_count=int(public_metrics.get("reply_count", row.get("reply_count", 0)) or 0),
            favorite_count=int(public_metrics.get("like_count", row.get("favorite_count", 0)) or 0),
            num_hashtags=_count_items(entities.get("hashtags", 0)),
            num_urls=_count_items(entities.get("urls", 0)),
            num_mentions=_count_items(entities.get("user_mentions", 0)),
            timestamp=_parse_timestamp(row.get("timestamp", row.get("created_at", 0))),
        )


@dataclass
class Sample:
    tweet_data: TweetData
    user_data: UserData
    label: str


class Cresci17SetTypes(Enum):
    GENUINE_USER = "human"  
    FAKE_FOLLOWER = "Fake_Follower"
    SOCIAL_SPAM_1 = "Social_Spam"
    SOCIAL_SPAM_2 = "Social_Spam"
    SOCIAL_SPAM_3 = "Social_Spam"
    TRADITIONAL_SPAM_1 = "Traditional_Spam"
    TRADITIONAL_SPAM_2 = "Traditional_Spam"
    TRADITIONAL_SPAM_3 = "Traditional_Spam"
    TRADITIONAL_SPAM_4 = "Traditional_Spam"

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