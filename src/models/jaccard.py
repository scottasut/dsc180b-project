import random
import pandas as pd
import sys
import logging
sys.path.append('../')
from src.util.logger_util import configure_logger
configure_logger('../log.txt')
log = logging.getLogger(__name__)

class SimpleJaccard:
    def __init__(self, data) -> None:
        """Instantiates and fills data structures for making Jaccard similarity recommendations.

        Args:
            data (collection): collection containing tuples of two elements: (user, subreddit). Pandas.DataFrame acceptable.

        Raises:
            ValueError: input data does not conform to expected format.
        """
        log.info('model instantiation entry.')
        if type(data) == pd.DataFrame:
            data = data.to_numpy()
        self._interacted_in = {}
        self._users = set()
        for d in data:
            if len(d) != 2:
                log.exception('initialization attempt with unexpected input data. Expected: (user, subreddit), Got: {}'.format(d))
                raise ValueError('Elements of \'data\' should be tuples of the following format: (user, subreddit)')
            u, sr = d
            self._users.add(u)
            if u not in self._interacted_in:
                self._interacted_in[u] = {}
            if sr not in self._interacted_in[u]:
                self._interacted_in[u][sr] = 1
            else:
                self._interacted_in[u][sr] += 1
        log.info('model instantiation exit.')

    def recommend(self, user: str, n=1) -> list:
        """Recommends n subreddits to a given user that they are not a part of using Jaccard similariy of other users as metric for fit likelihood.

        Args:
            user (str): name of user to make recommendation for
            n (int, optional): Number of recommendations to make. If less than n exist, returns maximum possible. Defaults to 1.

        Raises:
            ValueError: n non-positive

        Returns:
            list: subreddit recommendations
        """
        log.info('recommend entry for user {}, n={}'.format(user, n))
        if n <= 0:
            log.exception('recommendation attempt with non-positive output cardinality n={}.'.format(n))
            raise ValueError('\'n\' must be a positive integer.')
        others = []
        for other in self._users:
            if other == user:
                continue
            coef = self._coef(user, other)
            if coef < 1:
                others.append((coef, other))
        others.sort(reverse=True)
        subreddits = self._interacted_in[user].keys()
        recommendations = set()
        for coef, other in others:
            if len(recommendations) >= n:
                break
            for osr in self._interacted_in[other].keys():
                if osr not in subreddits:
                    recommendations.add(osr)
        recommendations = tuple(recommendations)
        if n == 1:
            out = [random.choice(recommendations)]
            log.info('recommend exit for user {}, n=1. Output: {}'.format(user, out))
            return out
        else:
            out = random.sample(recommendations, k=min(n, len(recommendations)))
            log.info('recommend exit for user {}, n={}. Output: {}'.format(user, n, out))
            return out

    def _coef(self, u1: str, u2: str) -> float:
        """Calculates the Jaccard coefficient between two users based off of given input data.

        Args:
            u1 (str): first user.
            u2 (str): second user.

        Returns:
            float: Similarity coefficient between u1 and u2 based off of interacted subreddits. Value between 0 and 1, larger means more similar.
        """
        if u1 not in self._interacted_in:
            log.exception('_coef calculation between users {}, {} failed: user \'{}\' does not exist.'.format(u1, u2, u1))
            raise ValueError('user \'{}\' does not exist.'.format(u1))
        if u2 not in self._interacted_in:
            log.exception('_coef calculation between users {}, {} failed: user \'{}\' does not exist.'.format(u1, u2, u2))
            raise ValueError('user \'{}\' does not exist.'.format(u2))
        u1_inter, u2_inter = set(self._interacted_in[u1].keys()), set(self._interacted_in[u2].keys())
        return len(u1_inter.intersection(u1_inter)) / len(u1_inter.union(u2_inter))
