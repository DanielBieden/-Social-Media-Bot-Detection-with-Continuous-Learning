This is the repo for "Social Media Bot Detection with Continuous Learning" for the course "Natural Language Processing and Neural Language Models" from the University Ulm.


#DATASETS

There are currently 8 datasets_utils implemented. They can be divided into Iterable_Datasets or (classic) Datasets.

CLASSIC DATASETS:
Cresci19, Gilani17 and Midterm18 are implemented as classic Datasets. These datasets only contain information about the users.

PARAMETERS:
    Every dataset can have the root parameter, which can be the filepath to the folder in which everywhing was downloaded. 
    If no parameter was give, root = datasets.
    IMPORTANT: If you can create the folders first and directly download the datasets into them, it will shorten the time.

METHODS:
    Each one has a __len__ method taht returns the amount of user profiles in the dataset.
    Additionally, they have the __getitem__(user_id:int)  method that returns the user_meta_data as a dataframe for that user:
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

ITERABLE_DATASETS: 

Caverlee11, Cresci17, Cresc18, Twibot20 and Twibot22 are implemented as Iterable Datasets. 
Each dataset contains user information as well as tweet information to varying degrees. 

PARAMETERS:
    Every dataset can have the root parameter, which can be the filepath to the folder in which everywhing was downloaded. 
    If no parameter was give, root = datasets.
    :param mode: Dataset split to use ("train", "dev", or "test").
    :param train_split: Fraction of users assigned to the training set (e.g. 0.8 = 80%).
    :param dev_split: Fraction of users assigned to the validation set (e.g. 0.1 = 10).
    IMPORTANT: If you can create the folders first and directly download the datasets into them, it will shorten the time (ESCPECIALLY for the twibot datasets).

    IMPORTANT: Cresci17 has the necessary parameter of Cresci17SetTypes, which can only be  
        GENUINE_USER  
        FAKE_FOLLOWER 
        SOCIAL_SPAM_1 
        SOCIAL_SPAM_2 
        SOCIAL_SPAM_3
        TRADITIONAL_SPAM_1 
        TRADITIONAL_SPAM_2 
        TRADITIONAL_SPAM_3
        TRADITIONAL_SPAM_4 

METHOD:
    Through the __iter__ method each dataset returns a Sample consisting of

    tweet_data: TweetData
    user_data: UserData
    label: str

    TweetData is implemented through: 
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

    UserData is implemented through:
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

    label is implemented as:
        "bot" or "human" or , for unlabeled, ""
