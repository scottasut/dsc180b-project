import pandas as pd
from sklearn.neighbors import NearestNeighbors

class CosineKNN:
    
    def __init__(self, data: pd.DataFrame):
        '''
        Input data should be dataframe with schema
        +--------------------+
        | user  | subreddit  |
        +-------+------------+
        | user1 | subreddit1 |
        | user1 | subreddit2 |
        | user1 | subreddit3 |
        | user2 | subreddit1 |
        | user2 | subreddit3 |
        +--------------------+
        '''
        data.columns = ['user', 'subreddit']
        self._users = data['user'].unique()
        self._subs = data['subreddit'].unique()
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
    
    def recommend(self, subreddits, n=1):
        sub_idx = [self._subreddits_encoding[sr] for sr in subreddits]
        distances, indices = self._knn.kneighbors(self._mat.loc[sub_idx], n_neighbors=n + 1)
        result = {}
        for i, sr in enumerate(subreddits):
            dists = distances[i]
            idxs = indices[i]
            result[sr] = [(self._subreddits_reverse_encoding[rec], d) for rec, d in zip(idxs, dists)]
        return result
