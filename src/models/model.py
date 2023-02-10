from pyTigerGraph import TigerGraphConnection
import pyTigerGraph as tg
import json
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import random

class RedditGraph():
    def __init__(self, config_path):
        self.config_path = config_path
        self.conn = self.connection()
        self.f = self.conn.gds.featurizer()
        self.generate_secret()
    
    def connection(self):
            with open(self.config_path, "r") as config:
                args = json.load(config)
            
            conn = tg.TigerGraphConnection(
                host=args["host"],
                graphname=args["graphname"],
                username=args["username"],
                password=args["password"],
                gsqlSecret=args["gsqlSecret"],
                certPath=args["certPath"]
            )

            if not self.connected(conn):
                raise ConnectionError("Failed to connect to GSQL")
            return conn

    def connected(self, conn):
        if not conn.echo() == "Hello GSQL":
            return False
        return True

    def generate_secret(self):
        with open(self.config_path, "r") as config:
            args = json.load(config)

        self.conn.getToken(args['gsqlSecret'])

    def getTotalVertexCount(self):
        print(self.conn.getVertexCount('*'))

    def getTotalEdgeCount(self):
        print(self.conn.getEdgeCount())

    def pagerank(self, v_type, e_type, result_attr):
        tg_pagerank_params = {
            "v_type": v_type,
            "e_type": e_type,
            "result_attr": result_attr,
        }
        results = pd.json_normalize(self.f.runAlgorithm("tg_pagerank",tg_pagerank_params)[0]['@@top_scores_heap'])
        return results

    def degree_cent(self, v_type, e_type, result_attr):
        tg_degree_params = {
            "v_type": v_type,
            "e_type": e_type,
            "result_attr": result_attr
        }
        results = pd.json_normalize(self.f.runAlgorithm("tg_degree_cent",tg_degree_params)[0]['top_scores'])
        return results

    def closeness(self, v_type, e_type, max_hops=10, top_k=100):
        tg_closeness_params = {
            "v_type": v_type,
            "e_type": e_type,
            "max_hops": max_hops,
            "top_k": top_k
        }

        return self.f.runAlgorithm("tg_closeness_cent", params=tg_closeness_params, timeout=600000)

    def split_vertices(self, train=.9, test=.1):
        splitter = self.conn.gds.vertexSplitter(is_train=train, is_test=test)
        splitter.run()

    def split_edges(self, train=.9, test=.1):
        splitter = self.conn.gds.edgeSplitter(is_train=train, is_test=test)
        splitter.run()

    def get_top_cmts(self, user):
        # returns top 3 highest karma comments made by user
        parameters = {
            "u": user
        }
        results = self.conn.runInstalledQuery("get_top_cmts", params=parameters)
        return results

    def get_subreddits(self, user):
        # returns all subreddits a user has interacted in
        parameters = {
            "u": user
        }
        results = self.conn.runInstalledQuery("get_subreddits", params=parameters)
        subs = set()
        for comment in results[0]['comments']:
            subs.add(comment['attributes']['comments.subreddit'])
        return subs

    def get_neighboring_subreddits(self, user):
        # returns all of the subreddits neighboring users are apart of
        parameters = {
            "u": user
        }
        results = self.conn.runInstalledQuery("get_neighbor_subreddits", params=parameters)
        subs = set()
        for comment in results[0]['comments']:
            subs.add(comment['attributes']['comments.subreddit'])
        return subs

    def calc_user_score(self, user):
        # given a user, calculates how similar they are compared to neighboring users (users they've interacted with)
        # similarity metric between two users is determined by number of subreddits intersected / number of subreddits in union
        parameters = {
            "u": user
        }
        results = self.conn.runInstalledQuery("get_users_subs", params=parameters)

        my_subs = self.get_subreddits(user)
        ans = {}
        
        for comment in results[0]['users_subs']:
            other_subs = set(comment['attributes']['users_subs.@subs'])
            total_subs = len(my_subs.union(other_subs))
            score = len(my_subs.intersection(other_subs)) / total_subs
            ans[comment['v_id']] = score
            
        return ans

    def diff_subs(self, user_one, user_two):
        # given user one and two, return subreddits of user_two that user_one isnt apart of
        one_subs = self.get_subreddits(user_one)
        two_subs = self.get_subreddits(user_two)
        return two_subs - one_subs

    def calc_sim(self, sub1, sub2):
        # calculates similarity between two subreddits
        # number of intersecting users / number of users in union
        parameters = {
            "u": sub1,
            "z": sub2
        }

        result = self.conn.runInstalledQuery("calc_sim_subs_2", params=parameters)
        intersect = result[1]['@@user_intersect.size()']
        union = result[0]['@@user_union.size()']

        return (intersect / union)

    def pick_sub(self, user_one, user_two, n=3):
        # picks sub for user one using subreddit similarity score
        one_subs = self.get_subreddits(user_one)
        new_subs = self.diff_subs(user_one, user_two)

        scores = {}
        for sub1 in one_subs:
            for sub2 in new_subs:
                scores[sub1 + "+" + sub2] = self.calc_sim(sub1, sub2)

        rec_subs = set()
        for tup in sorted(scores.items(), key=lambda x:x[1])[-n:]:
            rec_subs.add(tup[0].split("+")[1])

        return rec_subs

    def recommend_subs(self, user, sim_score=2, num_rec=1):
        user_scores = self.calc_user_score(user)
        recs = set()

        threshold = sorted(user_scores.values())[-sim_score:][0]

        for neighbor_user, score in user_scores.items():
            if score >= threshold:
                recs.update(random.sample(self.pick_sub(user, neighbor_user), num_rec))

        return recs