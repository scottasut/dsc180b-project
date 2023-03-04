import os
import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import sys
sys.path.append('../')
from util.logger_util import configure_logger
sys.path.append('dataset')
log = logging.getLogger(__name__)
configure_logger('../../log.txt')

USER_TEMP_PATH      = 'data/temp/user.csv'
USER_OUT_PATH       = 'data/out/user.csv'
SUBREDDIT_TEMP_PATH = 'data/temp/subreddit.csv'
SUBREDDIT_OUT_PATH  = 'data/out/subreddit.csv'
WORD2VEC_MDL_PATH   = 'glove-wiki-gigaword-50'
WORD2VEC_MDL_SIZE   = 50

def extract_tfidf(corpus: pd.Series, n: int):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit_transform(corpus)
    feature_names = tfidf.get_feature_names()
    tfidf_scores = tfidf.idf_
    user_tfidf = []
    for user_comments in corpus:
        user_tfidf.append(tfidf.transform([user_comments]))
    user_keywords = []
    for i, user_tfidf_matrix in enumerate(user_tfidf):
        feature_index = user_tfidf_matrix.nonzero()[1]
        tfidf_scores = zip(feature_index, [user_tfidf_matrix[0, i] for i in feature_index])
        top_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:n]
        user_keywords.append(' '.join([feature_names[k] for k, s in top_keywords]))
    return user_keywords

def word2vec_embedding(model, keywords, expected_length):
    """Get the embeddings of a list of words from a pre-trained word2vec model.
    """
    keywords = keywords.split()
    embeddings = []
    vocab = set(model.key_to_index.keys())
    for word in keywords:
        if word in vocab:
            embeddings.append(model[word])
        else:
            embeddings.append(np.zeros(model.vector_size))
    result = np.array(embeddings).flatten()
    return result if len(result) == expected_length else np.pad(result, (0, expected_length - len(result)))

def load_word2vec():
    return api.load(WORD2VEC_MDL_PATH)

def generate_features(n_keywords=10) -> None:
    """Generates features from temporary data

    The only feature(s) we generate are keywords from a user's corpus.
    """

    log.info('Feature generation task entry.')

    print('Generating model features...')

    if not os.path.exists(USER_TEMP_PATH):
        raise FileNotFoundError('Unable to find needed file {}.'.format(USER_TEMP_PATH))
    if not os.path.exists(SUBREDDIT_TEMP_PATH):
        raise FileNotFoundError('Unable to find needed file {}.'.format(SUBREDDIT_TEMP_PATH))

    expected_length = n_keywords * WORD2VEC_MDL_SIZE
    w2v = load_word2vec()

    users = pd.read_csv(USER_TEMP_PATH, header=None)
    users.columns = ['user', 'corpus']
    users['corpus'] = users['corpus'].str.replace('[^a-zA-Z0-9\s]', '', regex=True)
    users['corpus'] = extract_tfidf(users['corpus'], n_keywords)
    users['embedding'] = users['corpus'].apply(lambda x: list(word2vec_embedding(w2v, x, expected_length)))
    user_embeddings = pd.DataFrame(users['embedding'].to_list())
    users = pd.concat([users['user'].to_frame(), user_embeddings], axis=1)

    subreddits = pd.read_csv(SUBREDDIT_TEMP_PATH, header=None)
    subreddits.columns = ['subreddit', 'description']
    subreddits['description'] = subreddits['description'].str.replace('[^a-zA-Z0-9\s]', '', regex=True)
    subreddits['description'] = extract_tfidf(subreddits['description'].values.astype('U'), n_keywords)
    subreddits['embedding'] = subreddits['description'].apply(lambda x: list(word2vec_embedding(w2v, x, expected_length)))
    subreddit_embeddings = pd.DataFrame(subreddits['embedding'].to_list())
    subreddits = pd.concat([subreddits['subreddit'].to_frame(), subreddit_embeddings], axis=1)

    users.to_csv(USER_OUT_PATH, index=False, header=False)
    subreddits.to_csv(SUBREDDIT_OUT_PATH, index=False, header=False)

    print('Feature generation complete.')

    log.info('Feature generation task exit.')
