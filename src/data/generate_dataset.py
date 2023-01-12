import os
import json
import praw
import kaggle

def prepare():
    if not os.path.exists('../../data'):
        os.mkdir('../../data')
        os.mkdir('../../data/raw')
        os.mkdir('../../data/temp')
        os.mkdir('../../data/out')

def download_kaggle():
    # Downloading Kaggle data. Requires Kaggle API config--please refer to README
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('timschaum/subreddit-recommender', '../../data/raw', unzip=True)

def get_nodes_set():
    users, subreddits = set(), set()
    with open('../../data/raw/reddit_user_data_count.csv', 'r') as f:
        for l in f.readlines():
            if l.startswith('user,subreddit,count'):
                continue
            user, subreddit, _ = l.split(',')
            users.add(user)
            subreddits.add(subreddit)
    return users, subreddits

def download_reddit():
    with open('../../config.json') as f:
        config = json.load(f)

    reddit = praw.Reddit(
        client_id=config['reddit_client_id'],
        client_secret=config['reddit_secret_token'],
        username=config['reddit_username'],
        password=config['reddit_password'],
        user_agent=config['reddit_user_agent']
    )
