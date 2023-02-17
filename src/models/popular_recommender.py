import pandas as pd
import sys
import logging
sys.path.append('../')
from src.util.logger_util import configure_logger
configure_logger('../log.txt')
log = logging.getLogger(__name__)

class PopularRecommender:
    """
    The most basic item (community) recommender: simply recommend the most popular items (subreddits)
    that a user does not already belong to
    """
    def __init__(self, data) -> None:
        """Instantiates and fills data structures for making simple popularity-based subreddit recommendations.

        Args:
            data (iterable): iterable containing tuples of two elements: (user, subreddit). Pandas.DataFrame acceptable.
        
        Raises:
            ValueError: input data does not conform to expected format.
        """
        log.info('model instantiation entry.')
        if type(data) == pd.DataFrame:
            data = data.to_numpy()
        self._interacted_in = {}
        self._users = set()
        subreddit_counts = {}
        for d in data:
            if len(d) != 2:
                log.exception('initialization attempt with unexpected input data. Expected: (user, subreddit), Got: {}'.format(d))
                raise ValueError('Elements of \'data\' should be tuples of the following format: (user, subreddit)')
            u, sr = d
            self._users.add(u)
            if u not in self._interacted_in:
                self._interacted_in[u] = set()
            self._interacted_in[u].add(sr)
            if sr not in subreddit_counts:
                subreddit_counts[sr] = 1
            else:
                subreddit_counts[sr] += 1
        self._popular_subreddits = [sr for _, sr in sorted([(v, k) for k, v in subreddit_counts.items()], reverse=True)]
        log.info('model instantiation exit.')
    
    def recommend(self, user: str, n=1, avoid=None) -> list:
        """_summary_

        Args:
            user (str): user to make recommendation for
            n (int, optional): number of subreddits to recommend. Defaults to 1.

        Raises:
            ValueError: passed user does not exist in model.

        Returns:
            list: recommendations
        """
        log.info('recommend entry for user {}, n={}'.format(user, n))
        if avoid == None:
            avoid = []
        if user not in self._interacted_in:
            log.exception('recommendation for user {} failed: user {} does not exist'.format(user, user))
            raise ValueError('user {} does not exist.'.format(user))
        user_subs = self._interacted_in[user]
        result = []
        for sr in self._popular_subreddits:
            if len(result) == n:
                log.info('recommend exit for user {}, n={}. Output: {}'.format(user, n, result))
                return result
            if sr not in user_subs and sr not in avoid:
                result.append(sr)
        log.info('recommend exit for user {}, n={}. Output: {}'.format(user, n, result))
        return result