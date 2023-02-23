import os
import pandas as pd
import nltk
import numpy as np
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt', quiet=True)
from rake_nltk import Rake
import yake
import logging
import sys
sys.path.append('../')
from util.logger_util import configure_logger
sys.path.append('dataset')
log = logging.getLogger(__name__)
configure_logger('../../log.txt')

USER_TEMP_PATH = 'data/temp/user.csv'
USER_OUT_PATH = 'data/out/user.csv'
WORD2VEC_MDL_PATH = 'glove-wiki-gigaword-50'
WORD2VEC_MDL_SIZE = 50

VALID_EXTRACTORS = ['keybert', 'rake', 'yake']

def extract_keywords(text: str, n: int, how: str) -> list:

    log.info('Keyword extraction task entry using {} with n={}'.format(how, n))

    if how.lower() not in VALID_EXTRACTORS:
        raise ValueError('Argument \'how\' must be one of {}'.format(VALID_EXTRACTORS))

    # TODO: add KeyBERT
    # def bert_extractor():
    #     raise NotImplementedError()
    
    def rake_extractor():
        r = Rake(min_length=1, max_length=1)
        r.extract_keywords_from_text(text)
        ranked = r.get_ranked_phrases()[:n]
        # ranked = pd.Series(ranked).unique()
        return ' '.join(ranked)
    
    def yake_extractor():
        keywords = yake.KeywordExtractor(n=1, windowsSize=3, top=n).extract_keywords(text)
        results = []
        for scored_keywords in keywords:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return ' '.join(results)

    log.info('Keyword extraction task entry using {} with n={}'.format(how, n))

    # TODO: Add KeyBERT
    if how == 'rake':
        return rake_extractor()
    else:
        return yake_extractor()

def tfidf(corpus: pd.Series, n: int):
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

def generate_features(n_keywords=10, how='tfidf') -> None:
    """Generates features from temporary data

    The only feature we generate are keywords from a user's corpus.
    """

    log.info('Feature generation task entry.')

    print('Generating model features...')

    if not os.path.exists(USER_TEMP_PATH):
        raise FileNotFoundError('Unable to find needed file {}.'.format(USER_TEMP_PATH))

    users = pd.read_csv(USER_TEMP_PATH)
    users.columns = ['user', 'corpus']
    users['corpus'] = users['corpus'].str.replace('[^a-zA-Z0-9\s]', '', regex=True)
    if how == 'tfidf':
        users['corpus'] = tfidf(users['corpus'], n_keywords)
    else:
        users['corpus'] = users['corpus'].apply(lambda x: extract_keywords(x, n_keywords, how))
    expected_length = n_keywords * WORD2VEC_MDL_SIZE
    w2v = load_word2vec()
    users['embedding'] = users['corpus'].apply(lambda x: list(word2vec_embedding(w2v, x, expected_length)))
    embeddings = pd.DataFrame(users['embedding'].to_list())
    users = pd.concat([users['user'].to_frame(), embeddings], axis=1)
    users.to_csv(USER_OUT_PATH, index=False, header=False)

    print('Feature generation complete.')

    log.info('Feature generation task exit.')
