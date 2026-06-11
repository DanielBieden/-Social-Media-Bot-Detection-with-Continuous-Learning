import ast
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import html
import re




    

def clean_text(t):
    if not t:
        return ""

    t = str(t)

    t = html.unescape(t)

    try:
        t = t.encode("utf-8").decode("unicode_escape")
    except Exception:
        pass

    t = t.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    t = re.sub(r"\s+", " ", t)
    t = t.lower()

    return t.strip()

def normalize_user_id(value):
    text = str(value or "").strip()
    if not text:
        return "None"
    return text if text.startswith("u") else f"u{text}"


def _count_items(value):
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


def _safe_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def _as_int(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if value in (None, ""):
        return 0
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return int(text == "true")
    try:
        return int(float(text))
    except (TypeError, ValueError):
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
        profile = _safe_dict(row.get("profile"))
        return UserData(
            id=normalize_user_id(row.get("id", 0)),
            name=row.get(profile.get("name")
                                        if profile and "name" in profile
                                        else row.get("name", None)),
            screen_name=row.get(profile.get("username")
                                        if profile and "username" in profile
                                        else row.get("screen_name", None)),
            statuses_count=_as_int(row.get("statuses_count", 0)),
            followers_count=_as_int(profile.get("followers_count")
                                        if profile and "followers_count" in profile
                                        else row.get("followers_count", 0)),
            friends_count=_as_int(profile.get("following_count")
                                        if profile and "following_count" in profile
                                        else row.get("friends_count", 0)),
            favourites_count=_as_int(row.get("favourites_count", 0)),
            listed_count=_as_int(profile.get("listed_count")
                                        if profile and "listed_count" in profile
                                        else row.get("listed_count", 0)),
            lang=row.get("lang", None),
            verified=_as_int(row.get("verified", 0)),
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
        public_metrics = _safe_dict(row.get("public_metrics"))
        entities = _safe_dict(row.get("entities"))
        text = clean_text(row.get("text", ""))

        return TweetData(
            id=str(row.get("id", 0) or row.get("tweet_id",0) or 0),
            text=clean_text(str(row.get("text", "") or row.get("tweet","") or None)),
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
            timestamp=_parse_timestamp(row.get("timestamp") or row.get("created_at"))
        )


@dataclass
class Sample:
    tweet_data: list[TweetData]
    user_data: UserData
    label: str


class Cresci17SetTypes(Enum):
    GENUINE_USER = "genuine_user"  
    FAKE_FOLLOWER = "fake_followers"
    SOCIAL_SPAM_1 = "social_spambots_1"
    SOCIAL_SPAM_2 = "social_spambots_2"
    SOCIAL_SPAM_3 = "social_spambots_3"
    TRADITIONAL_SPAM_1 = "traditional_spambots_1"
    TRADITIONAL_SPAM_2 = "traditional_spambots_2"
    TRADITIONAL_SPAM_3 = "traditional_spambots_3"
    TRADITIONAL_SPAM_4 = "traditional_spambots_4s"

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