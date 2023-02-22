import os
import pandas as pd
# from keybert import KeyBERT
import nltk
nltk.download('punkt');
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
    if how.lower() not in VALID_EXTRACTORS:
        raise ValueError('Argument \'how\' must be one of {}'.format(VALID_EXTRACTORS))

    # TODO: configure BERT?
    # def bert_extractor():
    #     bert = KeyBERT()
    #     keywords = bert.extract_keywords(text, keyphrase_ngram_range=(1, 1), top_n=n)
    #     results = []
    #     for scored_keywords in keywords:
    #         for keyword in scored_keywords:
    #             if isinstance(keyword, str):
    #                 results.append(keyword)
    #     return ' '.join(results)
    
    def rake_extractor():
        r = Rake()
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

    # if how == 'keybert':
    #     return bert_extractor()
    # el
    if how == 'rake':
        return rake_extractor()
    else:
        return yake_extractor() 

def generate_features() -> None:
    """Generates features from temporary data

    The only feature we generate are keywords from a user's corpus.
    """

    if not os.path.exists(USER_TEMP_PATH):
        raise FileNotFoundError('Unable to find needed file {}.'.format(USER_TEMP_PATH))

    users = pd.read_csv(USER_TEMP_PATH)
    users.columns = ['user', 'corpus']
    users['corpus'] = users['corpus'].apply(lambda x: extract_keywords(x, 10, 'rake'))
    users.to_csv(USER_OUT_PATH, index=False, header=False)
