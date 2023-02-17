

class Evaluator:

    def __init__(self, interaction_path: str):
        self._interactions = {}
        with open(interaction_path) as f:
            for line in f.readlines():
                user, subreddit = line.split(',')
                subreddit = subreddit.strip()
                if user not in self._interactions:
                    self._interactions[user] = set()
                self._interactions[user].add(subreddit)
    
    def precision(self, user: str, recommendations: list):
        """Calculates precision @ k where k is the length of the passed recommendations

        Args:
            user (str): _description_
            recommendations (list): _description_

        Returns:
            _type_: _description_
        """
        relevant_recs = self._interactions[user].intersection(set(recommendations))
        print(self._interactions[user])
        print(relevant_recs)
        return len(relevant_recs) / len(recommendations)

    def recall(self, user: str, recommendations: list):
        """Calculates recall @ k where k is the length of the passed recommendations

        Args:
            user (str): _description_
            recommendations (list): _description_

        Returns:
            _type_: _description_
        """
        relevant = self._interactions[user]
        relevant_recs = relevant.intersection(set(recommendations))
        return len(relevant_recs) / len(relevant)
    
    def r_precision(self):
        pass
