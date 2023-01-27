import zstandard as zstd
import json
import os

USER_PATH = 'data/out/users.csv'
COMMENT_PATH = 'data/out/comments.csv'
SUBREDDIT_PATH = 'data/out/subreddits.csv'
USER_COMMENT_PATH = 'data/out/users_comments.csv'
COMMENT_COMMENT_PATH = 'data/out/comments_comments.csv'
SUBREDDIT_COMMENT_PATH = 'data/out/subreddits_comments.csv'

REMOTE_DATA_PATH = 'https://files.pushshift.io/reddit/comments/RC_2015-12.zst'
LOCAL_DATA_PATH = 'data/raw/RC_2015-12.zst'
DELETED_USER = '[deleted]'

def prepare():
    if not os.path.exists('data'):
        os.mkdir('data')
        os.mkdir('data/raw')
        os.mkdir('data/out')

def download():
    prepare()

    print('Downloading data...')

    if os.path.exists(LOCAL_DATA_PATH):
        print('{} already exists, skipping download of {}'.format(LOCAL_DATA_PATH, REMOTE_DATA_PATH))
        return

    print('Downloading {}...'.format(REMOTE_DATA_PATH))
    os.system('wget -O {} {}'.format(LOCAL_DATA_PATH, REMOTE_DATA_PATH))

def build_graph():

    download()

    print('Building network...')

    subreddits = set()
    users = set()

    with open(USER_PATH, 'a') as uf, open(COMMENT_PATH, 'a') as cf, open(SUBREDDIT_PATH, 'a') as sf, open(USER_COMMENT_PATH, 'a') as ucf, open(COMMENT_COMMENT_PATH, 'a') as ccf, open(SUBREDDIT_COMMENT_PATH, 'a') as scf:
        with open(LOCAL_DATA_PATH, 'rb') as f:
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
                        
                        cid = comment['id']
                        author = comment['author']
                        subreddit = comment['subreddit']
                        created_utc = comment['created_utc']
                        karma = comment['score']
                        pid = comment['parent_id']

                        if author.lower() == DELETED_USER:
                            continue

                        if author not in users:
                            uf.write('{}\n'.format(author))
                        cf.write('{},{},{},{}\n'.format(cid, subreddit, created_utc, karma))
                        ucf.write('{},{}\n'.format(author, cid))
                        if pid.startswith('t1_'):
                            ccf.write('{},{}\n'.format(pid[3:], cid))
                        if subreddit not in subreddits:
                            sf.write('{}\n'.format(subreddit))
                        scf.write('{},{}\n'.format(subreddit, cid))
                        
                        users.add(author)
                        subreddits.add(subreddit)

                    previous_line = lines[-1]
