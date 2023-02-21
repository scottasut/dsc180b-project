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

class CosineKNN:
    """
    Simple model using The KNN algorithm on cosine similarities of Collaborative filtering features.
    """
    
    def __init__(self, data) -> None:
        """Instantiates and fills data structures for making KNN Cosine similarity recommendations. Trains sklearn.neighbors.NearestNeighbors model.

        Args:
            data (collection): collection containing tuples of two elements: (user, subreddit). Pandas.DataFrame acceptable.

        Raises:
            ValueError: input data does not conform to expected format.
        """
        log.info('model instantiation entry.')
        if not type(data) == pd.DataFrame:
            data = pd.DataFrame(data, columns = ['user', 'subreddit'])
        data.columns = ['user', 'subreddit']
        self._data = data
        self._users = data['user'].unique()
        self._subs = data['subreddit'].unique()
        self._user_sub_map = data.groupby('user')['subreddit'].apply(set).to_dict()
        data = data.groupby(['user', 'subreddit']).size().reset_index(name='n_comments')
        data_max = data.groupby(['user'])['n_comments'].max().reset_index(name='max_comments')
        data = pd.merge(data, data_max, on='user', how='left')
        data['rating'] = data['n_comments'] / data['max_comments']
        self._users_encoding = {u:i for i, u in enumerate(set(data['user']))}
        self._subreddits_encoding = {sr:i for i, sr in enumerate(set(data['subreddit']))}
        self._subreddits_reverse_encoding = {v:k for k, v in self._subreddits_encoding.items()}
        data['user'] = data['user'].map(self._users_encoding)
        data['subreddit'] = data['subreddit'].map(self._subreddits_encoding)
        data = data[['user', 'subreddit', 'rating']]
        self._mat = data.pivot(index='subreddit', columns='user', values='rating').fillna(0)
        self._knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        self._knn.fit(self._mat)
        log.info('model instantiation exit.')
    
    def subreddit_recommend(self, subreddits, n=1) -> dict:
        """Gets similar subreddits for each passed subreddit.

        Args:
            subreddits (collection): collection of subreddits to get similar subreddits for.
            n (int, optional): Number of other subreddits to recommend. Defaults to 1.

        Returns:
            dict: mapping of passed subreddits to most similar subreddits and similarity scores
        """
        log.info('subreddit recommend entry for subreddits={}, n={}'.format(subreddits, n))
        sub_idx = [self._subreddits_encoding[sr] for sr in subreddits]
        distances, indices = self._knn.kneighbors(self._mat.loc[sub_idx], n_neighbors=n + 1)
        result = {}
        for i, sr in enumerate(subreddits):
            dists = distances[i]
            idxs = indices[i]
            result[sr] = [(self._subreddits_reverse_encoding[rec], d) for rec, d in zip(idxs, dists)]
        log.info('subreddit recommend exit for subreddits={}, n={}. Output: {}'.format(subreddits, n, result))
        return result
    
    def recommend(self, user, n=1, top=5) -> list:
        """Recommends a subreddit to a user.

        Args:
            user (str): name of user to make recommendation for
            n (int, optional): number of subreddits to recommend. Defaults to 1.
            top (int, optional): number of other subreddits to recommend for each of users subreddits. Defaults to 5.

        Returns:
            list: Subreddit recommendations
        """
        log.info('recommend entry for user {}, n={}, top={}'.format(user, n, top))
        # user_data = self._data.loc[self._data['user'] == user]
        # if len(user_data) == 0:
        #     raise ValueError('user {} does not exist.'.format(user))
        # ratios = user_data['subreddit'].value_counts().sort_index() / self._data['subreddit'].value_counts().sort_index()
        # top_subreddits = ratios.dropna().sort_values(ascending=False).iloc[0:top]
        # return self.subreddit_recommend(top_subreddits.index, n=10)
        user_subs = self._user_sub_map[user]
        recs = self.subreddit_recommend(user_subs, n=top)
        to_rec = {}
        for sr, score in [item for sublist in recs.values() for item in sublist]:
            if sr in user_subs:
                continue
            if sr in to_rec:
                to_rec[sr] += score
            else:
                to_rec[sr] = score
        out = sorted(to_rec, key=to_rec.get)[:n]
        log.info('recommend exit for user {}, n={}, top={}. Output: {}'.format(user, n, top, out))
        return out