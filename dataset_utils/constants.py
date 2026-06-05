import numpy as np
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
        'protected',
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