from random import sample

class Evaluator:

    def __init__(self, interaction_path: str, subset=-1):
        self._interactions = {}
        self.users = set()
        with open(interaction_path) as f:
            i = 0
            for line in f.readlines():
                if i == subset:
                    break
                user, subreddit = line.split(',')
                self.users.add(user)
                subreddit = subreddit.strip()
                if user not in self._interactions:
                    self._interactions[user] = set()
                self._interactions[user].add(subreddit)
                i += 1
        self.users = list(self.users)
    
    def precision(self, mdl, at: int, subset_size=-1,**mdl_params):
        """Calculates precision @ k where k is the length of the passed recommendations

        Args:
            user (str): _description_
            recommendations (list): _description_

        Returns:
            _type_: _description_
        """
        users = self.users
        if subset_size != -1:
            users = sample(users, min(subset_size, len(users)))
            
        p_vals = []
        for user in users:
            recommendations = mdl.recommend(user, n=at, **mdl_params)
            relevant = self._interactions[user]
            relevant_recs = relevant.intersection(set(recommendations))
            p_vals.append(len(relevant_recs) / len(recommendations))
        return sum(p_vals) / len(p_vals)

    def recall(self, mdl, at: int, subset_size=-1, **mdl_params):
        """Calculates recall @ k where k is the length of the passed recommendations

        Args:
            user (str): _description_
            recommendations (list): _description_

        Returns:
            _type_: _description_
        """
        users = self.users
        if subset_size != -1:
            users = sample(users, min(subset_size, len(users)))
            
        r_vals = []
        for user in users:
            recommendations = mdl.recommend(user, n=at,**mdl_params)
            relevant = self._interactions[user]
            relevant_recs = relevant.intersection(set(recommendations))
            r_vals.append(len(relevant_recs) / len(relevant))
        return sum(r_vals) / len(r_vals)
    
    def precision_recall(self, mdl, at: list, subset_size=-1, **mdl_params):
        users = self.users
        if subset_size != -1:
            users = sample(users, min(subset_size, len(users)))
        
        max_at = max(at)
        scores = {}
        for k in at:
            scores[k] = {}
            scores[k]['p_vals'] = []
            scores[k]['r_vals'] = []            

        for user in users:
            try:
                recommendations = mdl.recommend(user, n=max_at, **mdl_params)
                for k in at:
                    recs_at_k = recommendations[:k]
                    relevant = self._interactions[user]
                    relevant_recs = relevant.intersection(set(recs_at_k))
                    scores[k]['p_vals'].append(len(relevant_recs) / len(recs_at_k))
                    scores[k]['r_vals'].append(len(relevant_recs) / len(relevant))
            except:
                continue
        
        results = {}
        for k in at:
            key = '@{}'.format(k)
            p_vals, r_vals = scores[k]['p_vals'], scores[k]['r_vals']
            results[key] = (sum(p_vals) / len(p_vals), sum(r_vals) / len(r_vals))
        return results
