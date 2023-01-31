import random

class SimpleJaccard:
    def __init__(self, data: list):
        """
        'data' should be a list of tuples of the following format: (user, subreddit)
        """
        self._interacted_in = {}
        self._users = set()
        for d in data:
            if type(d) not in [tuple, list] or len(d) != 2:
                raise ValueError('Elements of \'data\' should be tuples of the following format: (user, comment, subreddit)')
            u, sr = d
            self._users.add(u)
            if u not in self._interacted_in:
                self._interacted_in[u] = {}
            if sr not in self._interacted_in[u]:
                self._interacted_in[u][sr] = 1
            else:
                self._interacted_in[u][sr] += 1

    def score():
        pass

    def reccomend(self, user, n=1):
        if n <= 0:
            raise ValueError('\'n\' must be a positive integer.')
        others = []
        for other in self._users:
            if other == user:
                continue
            others.append((self._coef(user, other), other))
        others.sort(reverse=True)
        for coef, other in others:
            if coef < 1:
                other_subreddits = set(self._interacted_in[other].keys())
                subreddits = set(self._interacted_in[user].keys())
                reccomendations = tuple(other_subreddits - subreddits)
                if n == 1:
                    return random.choice(reccomendations)
                else:
                    return random.sample(reccomendations, k=min(n, len(reccomendations)))

    def _coef(self, u1, u2):
        '''
        returns: Similarity coefficient between u1 and u2 based off of interacted subreddits.
                 Value between 0 and 1, larger means more similar.
        '''
        u1_inter, u2_inter = set(self._interacted_in[u1].keys()), set(self._interacted_in[u2].keys())
        return len(u1_inter.intersection(u1_inter)) / len(u1_inter.union(u2_inter))
