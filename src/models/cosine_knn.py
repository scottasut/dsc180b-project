import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sys
import logging
sys.path.append('../')
from src.util.logger_util import configure_logger
configure_logger('../log.txt')
log = logging.getLogger(__name__)

class CosineKNN:
    '''
    Simple model using The KNN algorithm on cosine similarities of Collaborative filtering feautures.
    '''
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
        """Gets simialar subreddits for each passed subreddit.

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
    
    def user_recommend(self, user, n=1, top=5) -> list:
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