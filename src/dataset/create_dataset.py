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

    print('Downloading data...')

    if os.path.exists(LOCAL_COMMENT_DATA_PATH):
        print('{} already exists, skipping download of {}'.format(LOCAL_COMMENT_DATA_PATH, REMOTE_COMMENT_DATA_PATH))
    else:
        print('Downloading {}...'.format(REMOTE_COMMENT_DATA_PATH))
        os.system('wget -O {} {}'.format(LOCAL_COMMENT_DATA_PATH, REMOTE_COMMENT_DATA_PATH))

    if os.path.exists(LOCAL_SUBREDDIT_DATA_PATH):
        print('{} already exists, skipping download of {}'.format(LOCAL_SUBREDDIT_DATA_PATH, REMOTE_SUBREDDIT_DATA_PATH))
    else:
        print('Downloading {}...'.format(REMOTE_SUBREDDIT_DATA_PATH))
        os.system('wget -O {} {}'.format(LOCAL_SUBREDDIT_DATA_PATH, REMOTE_SUBREDDIT_DATA_PATH))
    
    if os.path.exists(LOCAL_USER_DATA_PATH):
        print('{} already exists, skipping download of {}'.format(LOCAL_USER_DATA_PATH, REMOTE_USER_DATA_PATH))
    else:
        print('Downloading {}...'.format(REMOTE_USER_DATA_PATH))
        os.system('wget -O {} {}'.format(LOCAL_USER_DATA_PATH, REMOTE_USER_DATA_PATH))

def build_graph():

    # download()

    print('Building network...')

    subreddits = set()
    users = set()
    user_data = {}
    subreddit_data = {}
    comment_user_map = {}

    with lzma.open(LOCAL_USER_DATA_PATH) as f:
        for line in f:
            line = line.decode('UTF-8')
            data = json.loads(line)
            name = data['name']
            data.pop('name', None)
            user_data[name] = data
    
    for l in gzip.open(LOCAL_SUBREDDIT_DATA_PATH, 'rt'):
        data = json.loads(l)
        # attrs = set(['allow_images', 'allow_videogifs', 'allow_videos', 'created_utc', 'display_name', 'over18' 'public_description', 'subscribers'])
        # data = {k:v for k,v in data.items() if k in attrs}
        name = data['display_name']
        data.pop('display_name', None)
        subreddit_data[name] = data

    with open(USER_PATH, 'a') as uf, open(COMMENT_PATH, 'a') as cf, open(SUBREDDIT_PATH, 'a') as sf, open(USER_COMMENT_PATH, 'a') as ucf, open(COMMENT_COMMENT_PATH, 'a') as ccf, open(SUBREDDIT_COMMENT_PATH, 'a') as scf:
        with open(LOCAL_COMMENT_DATA_PATH, 'rb') as f:
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            with dctx.stream_reader(f) as reader:
                previous_line = ""
                while True:
                    chunk = reader.read(2**24)
                    if not chunk:
                        break

                    string_data = chunk.decode('utf-8')
                    lines = string_data.split("\n")
                    for i, line in enumerate(lines[:-1]):
                        if i == 0:
                            line = previous_line + line
                        comment = json.loads(line)
                        
                        # Comment Data
                        cid = comment['id']
                        author = comment['author']
                        subreddit_name = comment['subreddit']
                        created_utc = comment['created_utc']
                        karma = comment['score']
                        pid = comment['parent_id']
                        # contents = comment['body'].translate(None, string.punctuation).lower()

                        # Subreddit Data
                        subreddit = subreddit_data[subreddit_name]
                        allow_images = subreddit['allow_images']
                        allow_videogifs = subreddit['allow_videogifs']
                        allow_videos = subreddit['allow_videos']
                        sub_created_utc = subreddit['created_utc']
                        sub_over18 = subreddit['over18']
                        # public_description = subreddit['public_description'].translate(None, string.punctuation).lower()
                        subscribers = subreddit['subscribers']

                        # User Data
                        user = user_data['author']
                        link_karma = user['link_karma']
                        comment_karma = user['comment_karma']
                        profile_over_18 = user['profile_over_18']

                        if author.lower() == DELETED_USER:
                            continue

                        if author not in users:
                            uf.write('{},{},{},{}\n'.format(author, link_karma, comment_karma, profile_over_18))
                        cf.write('{},{},{},{}\n'.format(cid, subreddit_name, created_utc, karma))
                        ucf.write('{},{}\n'.format(author, cid))
                        comment_user_map[cid] = author
                        if pid.startswith('t1_'):
                            ccf.write('{},{}\n'.format(pid[3:], cid))
                        if subreddit_name not in subreddits:
                            sf.write('{},{},{},{},{},{},{}\n'.format(subreddit_name, allow_images, allow_videogifs, allow_videos, sub_created_utc, sub_over18, subscribers))
                        scf.write('{},{}\n'.format(subreddit_name, cid))
                        
                        users.add(author)
                        subreddits.add(subreddit_name)

                    previous_line = lines[-1]
    
    ccdf = pd.read_csv(COMMENT_COMMENT_PATH, header=['c1', 'c2'])
    ccdf['c1'] = ccdf['c1'].replace(comment_user_map)
    ccdf['c2'] = ccdf['c2'].replace(comment_user_map)
    uudf = ccdf.groupby(['c1', 'c2']).count()
    uudf.to_csv(USER_USER_PATH)

