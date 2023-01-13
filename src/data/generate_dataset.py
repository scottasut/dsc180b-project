import os
import json
import praw
import kaggle
import queue
import re

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
    return sorted(list(users)), sorted(list(subreddits))

def download_reddit():
    with open('../../configs/config.json') as f:
        config = json.load(f)

    reddit = praw.Reddit(
        client_id=config['reddit_client_id'],
        client_secret=config['reddit_secret_token'],
        username=config['reddit_username'],
        password=config['reddit_password'],
        user_agent=config['reddit_user_agent']
    )

    def get_comments_from_user(user, limit=None):
        return list(reddit.redditor(user).comments.new(limit=10))
    
    users, subreddits = get_nodes_set()
    users = users[:50]

    with open('../../data/raw/comments.csv', 'w') as cf:
        with open('../../data/raw/users.csv', 'w') as uf:
            with open('../../data/raw/users_comments.csv', 'w') as ucf:
                with open('../../data/raw/comments_comments.csv', 'w') as ccf:
                    for user in users:
                        try:
                            comments = get_comments_from_user(user)
                        except:
                            continue

                        uf.write('{}\n'.format(user))

                        for comment in comments:

                            # if comment.subreddit.name not in subreddits:
                            #     print('skipped')
                            #     continue

                            ucf.write('{},{}\n'.format(user, comment.id))

                            parent = comment
                            while type(parent.parent()) == reddit.comment:
                                parent = parent.parent()
                            
                            q = queue.Queue()
                            q.put(parent)
                            while not q.empty():
                                comment = q.get()
                                replies = list(comment.replies)
                                for reply in replies:
                                    q.put(reply)
                                
                                ccf.write('{},{}\n'.format(comment.id, comment.parent_id[3:]))
                                cf.write('{},{},{},{}\n'.format(comment.id, comment.created_utc, comment.score, comment.subreddit.display_name))
                                uf.write('{}\n'.format(comment.author.name))

                            cf.write('{},{},{},{}\n'.format(comment.id, comment.created_utc, comment.score, comment.subreddit.display_name))

download_reddit()
