import os
import pandas as pd
import nltk
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
        return ' '.join(r.get_ranked_phrases()[:n])
    
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

def generate_features(n_keywords=10, how='rake') -> None:
    """Generates features from temporary data

    The only feature we generate are keywords from a user's corpus.
    """

    log.info('Feature generation task entry.')

    print('Generating model features...')

    if not os.path.exists(USER_TEMP_PATH):
        raise FileNotFoundError('Unable to find needed file {}.'.format(USER_TEMP_PATH))

    users = pd.read_csv(USER_TEMP_PATH)
    users.columns = ['user', 'corpus']
    users['corpus'] = users['corpus'].apply(lambda x: extract_keywords(x, n_keywords, how))
    users.to_csv(USER_OUT_PATH, index=False, header=False)

    print('Feature generation complete.')

    log.info('Feature generation task exit.')
