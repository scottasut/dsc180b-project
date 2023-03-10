import zstandard as zstd
import json
import os
import logging
import sys
import pandas as pd
sys.path.append('../')
from util.logger_util import configure_logger
sys.path.append('dataset')
log = logging.getLogger(__name__)
configure_logger('../../log.txt')


USER_PATH = 'data/out/users.csv'
COMMENT_PATH = 'data/out/comments.csv'
SUBREDDIT_PATH = 'data/out/subreddits.csv'
USER_COMMENT_PATH = 'data/out/users_comments.csv'
COMMENT_COMMENT_PATH = 'data/out/comments_comments.csv'
USER_USER_PATH = 'data/out/users_users.csv'
SUBREDDIT_COMMENT_PATH = 'data/out/subreddits_comments.csv'
TEST_INTERACTION_PATH = 'data/out/test_interactions.csv'

REMOTE_COMMENT_DATA_PATH = 'https://files.pushshift.io/reddit/comments/RC_2010-12.zst'
REMOTE_TEST_COMMENT_DATA_PATH = 'https://files.pushshift.io/reddit/comments/RC_2011-12.zst'
REMOTE_SUBREDDIT_DATA_PATH = 'https://files.pushshift.io/reddit/subreddits/older_data/subreddits.json.gz'
REMOTE_USER_DATA_PATH = 'https://files.pushshift.io/reddit/authors/RA_2011.xz'
LOCAL_COMMENT_DATA_PATH = 'data/raw/RC_2010-12.zst'
LOCAL_TEST_COMMENT_DATA_PATH = 'data/raw/RC_2011-12.zst'
MAX_UTC_2010 = 1293839998
LOCAL_SUBREDDIT_DATA_PATH = 'data/raw/subreddits.json.gz'
LOCAL_USER_DATA_PATH = 'data/raw/RA_2011.xz'
DELETED_USER = '[deleted]'

def prepare():
    log.info('data directory preparation task entry.')
    if not os.path.exists('data'):
        log.info('Creating \'data\' directory.')
        os.mkdir('data')
        os.mkdir('data/raw')
        os.mkdir('data/out')
    else:
        log.info('\'data\' directory already exists. Skipping directory creation.')
    log.info('Data directory preparation task exit.')

def download(remote_path, local_path):
    prepare()

    log.info('data download task entry.')
    print('Downloading data...\n')

    if os.path.exists(local_path):
        log.info('{} already exists, skipping download of {}'.format(local_path, remote_path))
        print('{} already exists, skipping download of {}\n'.format(local_path, remote_path))
    else:
        log.info('downloading {}...'.format(remote_path))
        print('Downloading {}...'.format(remote_path))
        os.system('wget -O {} {}'.format(local_path, remote_path))
    
    log.info('data downloading task exit.')

def build_graph():

    log.info('graph generation task entry.')

    download(REMOTE_COMMENT_DATA_PATH, LOCAL_COMMENT_DATA_PATH)

    print('Reading Data...')

    cids = set()
    comments = []

    log.info('parsing {}'.format(LOCAL_COMMENT_DATA_PATH))
    with open(LOCAL_COMMENT_DATA_PATH, 'rb') as f:
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        with dctx.stream_reader(f) as reader:
            previous_line = ""
            while True:
                chunk = reader.read(2**24)
                if not chunk:
                    break

                string_data = chunk.decode('utf-8')
                lines = string_data.split('\n')
                for i, line in enumerate(lines[:-1]):
                    if i == 0:
                        line = previous_line + line
                    comment = json.loads(line)

                    cids.add(comment['id'])
                    comments.append({
                        'id': comment['id'],
                        'author': comment['author'],
                        'subreddit': comment['subreddit'],
                        'created_utc': comment['created_utc'],
                        'karma': comment['score'],
                        'pid': comment['parent_id']
                    })
                previous_line = lines[-1]
    
    print('Building Graph...')
    log.info('building graph.')

    cu_map = {}
    
    log.info('writing to [\'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\']'.format(USER_PATH, COMMENT_PATH, SUBREDDIT_PATH, USER_COMMENT_PATH, COMMENT_COMMENT_PATH, SUBREDDIT_COMMENT_PATH))
    with open(USER_PATH, 'a') as uf, open(COMMENT_PATH, 'a') as cf, open(SUBREDDIT_PATH, 'a') as sf, open(USER_COMMENT_PATH, 'a') as ucf, open(COMMENT_COMMENT_PATH, 'a') as ccf, open(SUBREDDIT_COMMENT_PATH, 'a') as scf:
        for comment in comments:
            
            if comment['pid'][3:] not in cids:
                continue
        
            # Skip [deleted] users
            if comment['author'].lower() == DELETED_USER:
                continue
            
            uf.write('{}\n'.format(comment['author']))
            # TODO: Add features to user
            # uf.write('{},{},{},{}\n'.format(author, link_karma, comment_karma, profile_over_18))
            
            cf.write('{},{},{},{}\n'.format(comment['id'], comment['subreddit'], comment['created_utc'], comment['karma']))
            ucf.write('{},{}\n'.format(comment['author'], comment['id']))
            ccf.write('{},{}\n'.format(comment['pid'][3:], comment['id']))
            sf.write('{}\n'.format(comment['subreddit']))

            scf.write('{},{}\n'.format(comment['subreddit'], comment['id']))
            # TODO: Add features to Subreddits
            # sf.write('{},{},{},{},{},{},{}\n'.format(subreddit_name, allow_images, allow_videogifs, allow_videos, sub_created_utc, sub_over18, subscribers))
            cu_map[comment['id']] = comment['author']

    uu_counts = {}
    with open(COMMENT_COMMENT_PATH) as ccf:
        for l in ccf.readlines():
            c1, c2 = l.split(',')
            try:
                pair = (cu_map[c1], cu_map[c2.strip()])
            except:
                continue
            if pair in uu_counts:
                uu_counts[pair] += 1
            else:
                uu_counts[pair] = 1

    log.info('Writing to [\'{}\']'.format(USER_USER_PATH))
    with open(USER_USER_PATH, 'w') as uuf:
        for pair in uu_counts:
            u1, u2 = pair
            count = -uu_counts[pair]
            uuf.write('{},{},{}\n'.format(u1, u2, count))  

    print('Done!')
    log.info('graph generation task exit.')
        
def build_test_set():

    log.info('test data generation task entry.')

    download(REMOTE_TEST_COMMENT_DATA_PATH, LOCAL_TEST_COMMENT_DATA_PATH)

    test_interactions = {}
    cids = set()

    # comments_to_subreddit = {}
    users_to_subreddit = pd.read_csv('')
    # with open(USER_COMMENT_PATH) as ucf, open(COMMENT_PATH) as cf:
    #     for l in cf.readlines():
    #         c, sr, _, _ = l.split(',')
    #         comments_to_subreddit[c] = sr

    #     for l in ucf.readlines():
    #         u, c = l.split(',')
    #         cids.add(u)
    #         if u not in users_to_subreddit:
    #             users_to_subreddit[u] = set()
    #         users_to_subreddit[u].add(comments_to_subreddit[c.strip()])
    
    log.info('parsing {}'.format(LOCAL_TEST_COMMENT_DATA_PATH))
    with open(LOCAL_TEST_COMMENT_DATA_PATH, 'rb') as f:
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        with dctx.stream_reader(f) as reader:
            previous_line = ""
            while True:
                chunk = reader.read(2**24)
                if not chunk:
                    break

                string_data = chunk.decode('utf-8')
                lines = string_data.split('\n')
                for i, line in enumerate(lines[:-1]):
                    if i == 0:
                        line = previous_line + line
                    comment = json.loads(line)
                    user = comment['author']
                    subreddit = comment['subreddit']

                    if user in cids:
                        if user not in test_interactions:
                            test_interactions[user] = set()
                        test_interactions[user].add(subreddit)
                        
                previous_line = lines[-1]
    
    log.info('writing to {}'.format(TEST_INTERACTION_PATH))
    with open(TEST_INTERACTION_PATH, 'w') as f:
        for user in test_interactions.keys():
            for subreddit in test_interactions[user]:
                if subreddit not in users_to_subreddit[user]:
                    f.write('{}, {}\n'.format(user, subreddit))
    
    log.info('test data generation task exit.')
