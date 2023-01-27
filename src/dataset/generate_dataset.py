import os
import json
import praw
import kaggle
import queue
import pandas as pd
import numpy as np

KAGGLE_DATA_PATH_REMOTE = 'timschaum/subreddit-recommender'
KAGGLE_DATA_PATH_LOCAL = '../../data/raw/reddit_user_data_count.csv'
STARTING_USERS_PATH = '../../data/raw/starting_users.csv'
USERS_DATA_PATH = '../../data/out/users.csv'
COMMENTS_DATA_PATH = '../../data/out/comments.csv'
USERS_COMMENTS_DATA_PATH = '../../data/out/users_comments.csv'
COMMENTS_COMMENTS_DATA_PATH = '../../data/out/comments_comments.csv'

def prepare():
    if not os.path.exists('../../data'):
        os.mkdir('../../data')
        os.mkdir('../../data/raw')
        os.mkdir('../../data/out')

def download_kaggle():
    # Downloading Kaggle data. Requires Kaggle API config--please refer to README
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_DATA_PATH_REMOTE, '../../data/raw', unzip=True)

def get_nodes_set():
    if not os.path.exists(KAGGLE_DATA_PATH_LOCAL):
        raise FileNotFoundError('File \'{}\' must exist. Call download_kaggle() first.'.format(KAGGLE_DATA_PATH_LOCAL))

    users, subreddits = set(), set()
    with open(KAGGLE_DATA_PATH_LOCAL, 'r') as f:
        for l in f.readlines():
            if l.startswith('user,subreddit,count'):
                continue
            user, subreddit, _ = l.split(',')
            users.add(user)
            subreddits.add(subreddit)
    return sorted(list(users)), sorted(list(subreddits))

def get_reddit():
    with open('../../configs/reddit_config.json') as f:
        config = json.load(f)

    return praw.Reddit(
        client_id=config['reddit_client_id'],
        client_secret=config['reddit_secret_token'],
        username=config['reddit_username'],
        password=config['reddit_password'],
        user_agent=config['reddit_user_agent']
    )

def download_reddit(reddit, num_users=100):
    with open('../../configs/reddit_config.json') as f:
        config = json.load(f)

    reddit = praw.Reddit(
        client_id=config['reddit_client_id'],
        client_secret=config['reddit_secret_token'],
        username=config['reddit_username'],
        password=config['reddit_password'],
        user_agent=config['reddit_user_agent']
    )

    def get_comments_from_user(user, limit=None):
        return list(reddit.redditor(user).comments.new(limit=None))
    
    users, subreddits = get_nodes_set()
    downloaded_users = set()
    read_users_in_batch = set()
    download_count = 0

    if os.path.exists(STARTING_USERS_PATH):
        with open(STARTING_USERS_PATH, 'r') as f:
            for l in f.readlines():
                downloaded_users.add(l.split()[0])

    with open(STARTING_USERS_PATH, 'a+') as suf, open(COMMENTS_DATA_PATH, 'a+') as cf, open(USERS_DATA_PATH, 'a+') as uf, open(USERS_COMMENTS_DATA_PATH, 'a+') as ucf, open(COMMENTS_COMMENTS_DATA_PATH, 'a+') as ccf:
        for user in users:

            if download_count == num_users:
                break

            # Skipping users we have previously read
            if user in downloaded_users:
                continue

            try:
                comments = get_comments_from_user(user)
            except:
                continue

            uf.write('{}\n'.format(user))
            suf.write('{}\n'.format(user))
            read_users_in_batch.add(user)

            for comment in comments:

                ucf.write('{},{}\n'.format(user, comment.id))

                parent = comment
                while type(parent.parent()) == reddit.comment:
                    parent = parent.parent()

                q = queue.Queue()
                try:
                    parent.refresh()
                except:
                    pass
                parent.replies.replace_more(limit=None)
                replies = parent.replies
                
                for reply in replies:
                    q.put(reply)
                while not q.empty():
                    comment = q.get()
                    replies = list(comment.replies)
                    for reply in replies:
                        q.put(reply)
                    
                    if comment.author:
                        ccf.write('{},{}\n'.format(comment.id, comment.parent_id[3:]))
                        cf.write('{},{},{},{}\n'.format(comment.id, comment.created_utc, comment.score, comment.subreddit.display_name))
                        ucf.write('{},{}\n'.format(comment.author.name, comment.id))
                        if comment.author.name not in read_users_in_batch and comment.author.name not in downloaded_users:
                            uf.write('{}\n'.format(comment.author.name))
                        read_users_in_batch.add(comment.author.name)

                cf.write('{},{},{},{}\n'.format(comment.id, comment.created_utc, comment.score, comment.subreddit.display_name))
            
            download_count += 1
            print("Users downloaded: {}".format(download_count))

prepare()
