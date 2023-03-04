import zstandard as zstd
import gzip
import json
import os
import pandas as pd
import logging
import sys
sys.path.append('../')
from util.logger_util import configure_logger
sys.path.append('dataset')
log = logging.getLogger(__name__)
configure_logger('../../log.txt')

# Pushshift.io hosts data for each month from Dec. 2005 to present (Jan. 2023)
# Files for corresponding months/years are given by:
# https://files.pushshift.io/reddit/comments/RC_<YEAR>-<MONTH>.zst
GENERIC_REMOTE_PATH = 'https://files.pushshift.io/reddit/comments/RC_{}-{}.zst'
GENERIC_LOCAL_PATH  = 'data/raw/RC_{}-{}.zst'

# Processed data paths:
USER_PATH              = 'data/temp/user.csv'             # User vertices
SUBREDDIT_PATH         = 'data/temp/subreddit.csv'        # Subreddit vertices
USER_USER_PATH         = 'data/out/user_user.csv'         # User to user edges
USER_SUBREDDIT_PATH    = 'data/out/user_subreddit.csv'    # User to subreddit edges
TEST_INTERACTIONS_PATH = 'data/out/test_interactions.csv' # User to subreddit interactions for testing

DELETED_USER = '[deleted]'

SUBREDDIT_KARMA_THRESHOLD      = 5
SUBREDDIT_N_COMMENTS_THRESHOLD = 25

def prepare() -> None:
    """Set up file structure to accommodate data
    """
    log.info('data directory preparation task entry.')
    if not os.path.exists('data'):
        log.info('Creating \'data\' directory.')
        os.mkdir('data')
        os.mkdir('data/raw')
        os.mkdir('data/temp')
        os.mkdir('data/out')
    else:
        log.info('\'data\' directory already exists. Skipping directory creation.')
    log.info('Data directory preparation task exit.')

def download(remote_path: str, local_path: str) -> None:
    """Download a remote file to project using wget

    Args:
        remote_path (str): remote path of data to download
        local_path (str): path to download file to
    """
    prepare()

    log.info('data download task entry for remote_path: {}, local_path: {}.'.format(remote_path, local_path))
    print('Downloading data...')

    if os.path.exists(local_path):
        log.info('{} already exists, skipping download of {}'.format(local_path, remote_path))
        print('{} already exists, skipping download of {}\n'.format(local_path, remote_path))
    else:
        log.info('downloading {}...'.format(remote_path))
        print('Downloading {}...'.format(remote_path))
        os.system('wget -O {} {}'.format(local_path, remote_path))
    
    log.info('data download task exit for remote_path: {}, local_path: {}.'.format(remote_path, local_path))

def process_data(year: str, month: str) -> None:
    """Download and process Reddit data for a given month/year into graph representation files.

    Args:
        year (str): Year of Reddit data
        month (str): Month of Reddit data
    """

    log.info('data processing task entry for year: {}, month: {}'.format(year, month))

    # Fetch the data
    remote_path = GENERIC_REMOTE_PATH.format(year, month)
    local_path = GENERIC_LOCAL_PATH.format(year, month)
    download(remote_path, local_path)

    # Process the data
    print('Processing data...')

    comment_user_map = {}
    user_data, subreddit_data = {}, {}
    user_interactions, subreddit_interactions = {}, {}
    subreddit_corpus_size = {}
    
    with open(local_path, 'rb') as f:
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

                    comment_id = comment['id']            # ID of the comment
                    user       = comment['author']        # Name of user who posted the comment
                    subreddit  = comment['subreddit']     # Name of Subreddit comment was posted to
                    parent_id  = comment['parent_id'][3:] # Comment ID of the parent of the comment
                                                          # (prefixed with irrelavant information)
                    karma = comment['score']              # net score of upvotes/downvotes on comment
                    comment_body = comment['body']        # Text contents of the comment
                    comment_body = comment_body.lower()

                    # When a user deletes their account, their comments remain
                    # and the author becomes '[deleted]'. We will skip these.
                    if user.lower() == DELETED_USER:
                        continue

                    comment_user_map[comment_id] = user

                    if user not in user_data:
                        user_data[user] = comment_body
                    else:
                        user_data[user] += ' ' + comment_body
                    
                    if subreddit not in subreddit_data:
                        subreddit_corpus_size[subreddit] = 0
                        subreddit_data[subreddit] = ''
                    
                    if karma >= SUBREDDIT_KARMA_THRESHOLD and subreddit_corpus_size[subreddit] <= SUBREDDIT_N_COMMENTS_THRESHOLD:
                        subreddit_data[subreddit] += ' ' + comment_body
                        subreddit_corpus_size[subreddit] += 1

                    # Some parent IDs reference a post rather than a comment,
                    # we will ignore these as interactions
                    if not parent_id.startswith('c'):
                        continue

                    user_interaction = tuple(sorted((comment_id, parent_id)))
                    if user_interaction not in user_interactions:
                        user_interactions[user_interaction] = 1
                    else:
                        user_interactions[user_interaction] += 1

                    subreddit_interaction = (user, subreddit)
                    if subreddit_interaction not in subreddit_interactions:
                        subreddit_interactions[subreddit_interaction] = 1
                    else:
                        subreddit_interactions[subreddit_interaction] += 1

                previous_line = lines[-1]
    
    users = pd.DataFrame(user_data.items())
    subreddits = pd.DataFrame(subreddit_data.items())
    users_users = pd.DataFrame(user_interactions.items())
    users_users = pd.concat([pd.DataFrame(users_users[0].to_list(), columns=['u1', 'u2']), users_users], axis=1).drop(0, axis=1)
    users_users['u1'] = users_users['u1'].map(comment_user_map)
    users_users['u2'] = users_users['u2'].map(comment_user_map)
    users_users.dropna(inplace=True)
    users_subreddits = pd.DataFrame(subreddit_interactions.items())
    users_subreddits = pd.concat([pd.DataFrame(users_subreddits[0].to_list(), columns=['u1', 'sr']), users_subreddits], axis=1).drop(0, axis=1)

    subreddits.to_csv(SUBREDDIT_PATH, index=False, header=False)
    users.to_csv(USER_PATH, index=False, header=False)
    users_users.to_csv(USER_USER_PATH, index=False, header=False)
    users_subreddits.to_csv(USER_SUBREDDIT_PATH, index=False, header=False)

    print('Processing data complete.')

    log.info('data processing task exit for year: {}, month: {}'.format(year, month))


def build_test_set(year: str, month: str) -> None:

    log.info('test data generation task entry.')

    # Fetch the data
    remote_path = GENERIC_REMOTE_PATH.format(year, month)
    local_path = GENERIC_LOCAL_PATH.format(year, month)
    download(remote_path, local_path)

    users_to_subreddit = pd.read_csv(USER_SUBREDDIT_PATH, header=None)
    users_to_subreddit.columns = ['user', 'subreddit', 'times']
    user_set = set(users_to_subreddit['user'])
    subreddit_set = set(users_to_subreddit['subreddit'])
    users_to_subreddit = users_to_subreddit.drop(columns=['times'])
    users_to_subreddit = users_to_subreddit.groupby('user').agg(lambda x: set(x))
    users_to_subreddit = users_to_subreddit.to_dict()['subreddit']
    user_subreddit = set()
    
    log.info('parsing {}'.format(local_path))
    with open(local_path, 'rb') as f, open(TEST_INTERACTIONS_PATH, 'w') as tif:
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

                    if (user, subreddit) in user_subreddit:
                        continue

                    if user in user_set and subreddit in subreddit_set and subreddit not in users_to_subreddit[user]:
                        tif.write('{},{}\n'.format(user, subreddit))
                        user_subreddit.add((user, subreddit))
                        
                previous_line = lines[-1]
