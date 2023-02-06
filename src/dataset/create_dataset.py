import zstandard as zstd
import lzma
import gzip
import json
import os
import pandas as pd

USER_PATH = 'data/out/users.csv'
COMMENT_PATH = 'data/out/comments.csv'
SUBREDDIT_PATH = 'data/out/subreddits.csv'
USER_COMMENT_PATH = 'data/out/users_comments.csv'
COMMENT_COMMENT_PATH = 'data/out/comments_comments.csv'
USER_USER_PATH = 'data/out/users_users.csv'
SUBREDDIT_COMMENT_PATH = 'data/out/subreddits_comments.csv'

REMOTE_COMMENT_DATA_PATH = 'https://files.pushshift.io/reddit/comments/RC_2010-12.zst'
REMOTE_SUBREDDIT_DATA_PATH = 'https://files.pushshift.io/reddit/subreddits/older_data/subreddits.json.gz'
REMOTE_USER_DATA_PATH = 'https://files.pushshift.io/reddit/authors/RA_2011.xz'
LOCAL_COMMENT_DATA_PATH = 'data/raw/RC_2010-12.zst'
LOCAL_SUBREDDIT_DATA_PATH = 'data/raw/subreddits.json.gz'
LOCAL_USER_DATA_PATH = 'data/raw/RA_2011.xz'
DELETED_USER = '[deleted]'

def prepare():
    if not os.path.exists('data'):
        os.mkdir('data')
        os.mkdir('data/raw')
        os.mkdir('data/out')

def download():
    prepare()

    print('Downloading data...\n')

    if os.path.exists(LOCAL_COMMENT_DATA_PATH):
        print('{} already exists, skipping download of {}\n'.format(LOCAL_COMMENT_DATA_PATH, REMOTE_COMMENT_DATA_PATH))
    else:
        print('Downloading {}...'.format(REMOTE_COMMENT_DATA_PATH))
        os.system('wget -O {} {}'.format(LOCAL_COMMENT_DATA_PATH, REMOTE_COMMENT_DATA_PATH))

    # TODO: Add data to subreddits and users
    # if os.path.exists(LOCAL_SUBREDDIT_DATA_PATH):
    #     print('{} already exists, skipping download of {}\n'.format(LOCAL_SUBREDDIT_DATA_PATH, REMOTE_SUBREDDIT_DATA_PATH))
    # else:
    #     print('Downloading {}...'.format(REMOTE_SUBREDDIT_DATA_PATH))
    #     os.system('wget -O {} {}'.format(LOCAL_SUBREDDIT_DATA_PATH, REMOTE_SUBREDDIT_DATA_PATH))
    
    # if os.path.exists(LOCAL_USER_DATA_PATH):
    #     print('{} already exists, skipping download of {}\n'.format(LOCAL_USER_DATA_PATH, REMOTE_USER_DATA_PATH))
    # else:
    #     print('Downloading {}...'.format(REMOTE_USER_DATA_PATH))
    #     os.system('wget -O {} {}'.format(LOCAL_USER_DATA_PATH, REMOTE_USER_DATA_PATH))

def build_graph():

    download()

    print('Reading Data...')

    cids = set()
    comments = []

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
    cu_map = {}

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

    with open(USER_USER_PATH, 'w') as uuf:
        for pair in uu_counts:
            u1, u2 = pair
            count = uu_counts[pair]
            uuf.write('{},{},{}\n'.format(u1, u2, count))

    print('Done!')
        

