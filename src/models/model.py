import pyTigerGraph as tg
import json
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from src.models.popular_recommender import PopularRecommender
import sys
import logging
sys.path.append('../')
from src.util.logger_util import configure_logger
configure_logger('../log.txt')
log = logging.getLogger(__name__)

GSQL_PATH_PREFIX = '/gsql/'

class RedditGraph():
    """Handler class for our TigerGraph data model and recommendation tasks.
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.conn = self.connection()
        self.f = self.conn.gds.featurizer()
        self.generate_secret()
        data = []
        comments_to_subreddit = {}
        with open('../data/out/users_comments.csv') as ucf, open('../data/out/comments.csv') as cf:
            for l in cf.readlines():
                c, sr, _, _ = l.split(',')
                comments_to_subreddit[c] = sr

            for l in ucf.readlines():
                u, c = l.split(',')
                data.append((u, comments_to_subreddit[c.strip()]))
        df = pd.DataFrame(data, columns=['user', 'subreddit'])
        self._popular_recommender = PopularRecommender(df)
    
    def connection(self):
        """Establishes a connetion to TigerGraph using credentials which should be in config.json

        Raises:
            ConnectionError: if the connection was not successful

        Returns:
            pyTiger: _description_
        """
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

        if not self._connected(conn):
            raise ConnectionError("Failed to connect to GSQL")
        return conn

    def _connected(self, conn: tg.TigerGraphConnection) -> bool:
        """Verifies a TigerGraph Connection is valid

        Args:
            conn (tg.TigerGraphConnection): _description_

        Returns:
            bool: whether connection is valid
        """
        if not conn.echo() == "Hello GSQL":
            return False
        return True

    def generate_secret(self):
        """generates a secret key for TigerGraph authentication.
        """
        with open(self.config_path, "r") as config:
            args = json.load(config)
        self.conn.getToken(args['gsqlSecret'])

    def get_total_vertex_count(self) -> int:
        """gets the number of vertices in the graph

        Returns:
            int: number of vertices in the graph
        """
        log.info('get_total_vertex_count entry')
        result = self.conn.getVertexCount('*')
        log.info('get_total_vertex_count exit. Output: {}'.format(result))
        return result

    def get_total_edge_count(self) -> int:
        """gets the number of edges in the graph

        Returns:
            int: number of edges in the graph
        """
        log.info('get_total_edge_count entry')
        result = self.conn.getEdgeCount()
        log.info('get_total_edge_count exit. Output: {}'.format(result))
        return result
    
    def get_vertex_type(self, v_type: str):
        log.info('get_vertex_type entry for v_type={}'.format(v_type))
        result = self.conn.getVertices(v_type)
        log.info('get_vertex_type exit for v_type={}. Output cardinality={}'.format(v_type, len(result)))
        return result
    
    def get_vertex_by_id(self, v_type: str, v_id: str):
        log.info('get_vertex_by_id entry for v_type={}'.format(v_type))
        result = self.conn.getVerticesById(v_type, [v_id])
        log.info('get_vertex_by_id exit for v_type={}. Output cardinality={}'.format(v_type, len(result)))
        return result

    def pagerank(self, v_type: str, e_type: str, result_attr: str):
        """runs the pagerank algorithm in TigerGraph for the given vertex and edges types.

        Args:
            v_type (str): vertex type to run pagerank for
            e_type (str): vertex type use in pagerank
            result_attr (str): name of result attribute on vertex

        Returns:
            list: algorithm result
        """
        tg_pagerank_params = {
            "v_type": v_type,
            "e_type": e_type,
            "result_attr": result_attr,
        }

        log.info('pagerank entry with params={}'.format(tg_pagerank_params))

        self.f.installAlgorithm('tg_pagerank')
        self._create_vertex_attr(v_type, result_attr)

        result = pd.json_normalize(self.f.runAlgorithm("tg_pagerank",tg_pagerank_params)[0]['@@top_scores_heap'])
        log.info('pagerank exit with params={}. Output={}'.format(tg_pagerank_params, result))
        return result

    def degree_cent(self, v_type: str, e_type: str, result_attr: str):
        """runs the degree centrality algorithm in TigerGraph for the given vertex and edges types.

        Args:
            v_type (str): vertex type to run degree centrality for
            e_type (str): vertex type use in degree centrality
            result_attr (str): name of result attribute on vertex

        Returns:
            list: algorithm result
        """
        tg_degree_params = {
            "v_type": v_type,
            "e_type": e_type,
            "result_attr": result_attr
        }

        log.info('degree_cent entry with params={}'.format(tg_degree_params))

        self.f.installAlgorithm('tg_degree_cent')
        self._create_vertex_attr(v_type, result_attr)

        result = pd.json_normalize(self.f.runAlgorithm("tg_degree_cent",tg_degree_params)[0]['top_scores'])
        log.info('degree_cent exit with params={}. Output={}'.format(tg_degree_params, result))
        return result

    def closeness(self, v_type: str, e_type: str, result_attribute: str, max_hops=10, top_k=100):
        """runs the vertex closeness algorithm in TigerGraph for the given vertex and edges types.

        Args:
            v_type (str): vertex type to run degree centrality for
            e_type (str): vertex type use in degree centrality
            result_attr (str): name of result attribute on vertex

        Returns:
            list: algorithm result
        """
        tg_closeness_params = {
            "v_type": v_type,
            "e_type": e_type,
            "max_hops": max_hops,
            "top_k": top_k
        }
        
        log.info('closeness entry with params={}'.format(tg_closeness_params))

        self.f.installAlgorithm('tg_closeness_cent')
        self._create_vertex_attr(v_type, result_attribute)

        log.info('closeness exit with params={}. Output={}'.format(tg_closeness_params, result))
        result = self.f.runAlgorithm("tg_closeness_cent", params=tg_closeness_params, timeout=600000)
        return result

    def louvain(self, v_type: list, e_type: list, wt_attr: str, result_attr: str, max_iter=10):
        """Runs the Louvain Algorithm in TigerGraph.

        Args:
            v_type (list): list of vertex types to run louvain for
            e_type (list): list of edge types to use for graph propagation
            wt_attr (str): name of the edge weight attribute
            result_attr (str): name of the result attribute on vertices
            max_iter (int, optional): Number of max iterations for algorithm. Defaults to 10.

        Returns:
            list: Algorithm result
        """
        louvain_params = {
            "v_type": v_type,
            "e_type": e_type,
            "wt_attr": wt_attr,
            "max_iter": max_iter,
            "result_attr": result_attr,
            "file_path": "",
            "print_info": False
        }

        log.info('louvain entry for params={}'.format(louvain_params))
        self.f.installAlgorithm('tg_louvain')
        for v in v_type:
            self._create_vertex_attr(v, result_attr)

        try:
            result = self.f.runAlgorithm('tg_louvain', params=louvain_params)
        except Exception as e:
            log.exception('failed to run louvain with params={}'.format(louvain_params, e))
            raise e
        
        log.info('louvain exit for params={}. Output: {}'.format(louvain_params, result))
        return result
    
    def predict_louvain(self, v_id: str, n=1):
        params = {"u": v_id}
        log.info('predict_louvain entry for params={}'.format(params))

        result = self.conn.runInstalledQuery('get_louvain_neighbors', params=params)
        neighbors = result[0]['neighbors']
        subreddit_pool = set()
        u_subs = self.get_subreddits(v_id)
        for u in neighbors:
            name = u['v_id']
            srs = self.get_subreddits(name)
            for sr in srs:
                subreddit_pool.add(sr)

        recommendations = list(subreddit_pool - u_subs)
        if len(recommendations) < n: # No difference in neighbors interactions
            pop_recs = self._popular_recommender.recommend(v_id, n=n - len(recommendations))
            for rec in pop_recs:
                recommendations.append(rec)
            log.info('predict_louvain exit for params={}, Output ={}'.format(params, recommendations))
            return rec
        else:
            recommendations = random.sample(recommendations, n)
            # similar_subs = {}
            # for s1 in u_subs:
            #     for s2 in eligible_subs:
            #         similar_subs[(s1, s2)] = self.calc_sim(s1, s2)
            
            # _, rec = max(similar_subs, key=similar_subs.get)

            log.info('predict_louvain exit for params={}, Output ={}'.format(params, recommendations))
            return recommendations
    
    def fit_knn(self, v_type: str, k=1):
        df = self.conn.getVertexDataFrame(v_type, select='embeddings')
        users = df['v_id']
        self._knn_user_idx_map = {i:u for i, u in zip(range(len(users)), users)}
        df = df['embeddings'].apply(pd.Series)
        df.columns = ['pagerank', 'louvain', 'label_prop', 'degree']
        self._knn_embeddings = df
        self._knn_embeddings.index = users
        self._knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k+1, n_jobs=-1).fit(df)

    def predict_knn(self, v_id: str, n=1):
        if not self._knn:
            raise Exception('Must call "fit_kmeans" first.')
        distances, indices = self._knn.kneighbors(np.array([self._knn_embeddings.loc[v_id]]))
        eligible_subs = set()
        u_subs = self.get_subreddits(v_id)
        for index in indices[0][1:]:
            other = self._knn_user_idx_map[index]
            other_subs = self.get_subreddits(other)
            for sr in other_subs:
                eligible_subs.add(sr)
        recommendations = list(eligible_subs - u_subs)
        if len(recommendations) > n:
            return recommendations[:n]
        else:
            while len(recommendations) < n:
                pop_recs = self._popular_recommender.recommend(v_id, n=n - len(recommendations), avoid=recommendations)
                for pr in pop_recs:    
                    recommendations.append(pr)
        return recommendations
    
    def recommend(self, user: str, n=1, how='knn'):
        how = how.lower()
        if how not in ['louvain', 'knn']:
            raise ValueError('\'how\' must be one of [\'louvain\', \'knn\']')

        if how == 'louvain':
            return self.predict_louvain(user, n=n)
        elif how == 'knn':
            return self.predict_knn(user, n)
    
    def label_propagation(self, v_type_set: list, e_type_set: list, attr: str, maximum_iteration=10):
        """Runs the Label Propagation Algorithm in TigerGraph.

        Args:
            v_type_set (list): list of vertex types to run Label Propagation for
            e_type_set (list): list of edge types to use for graph propagation
            attr (str): name of the edge weight attribute
            maximum_iteration (str): Number of max iterations for algorithm. Defaults to 10.

        Returns:
            list: Algorithm result
        """
        label_propagation_params = {
            "v_type_set": v_type_set,
            "e_type_set": e_type_set,
            "attr": attr,
            "maximum_iteration": maximum_iteration
        }

        log.info('label_propagation entry for params={}'.format(label_propagation_params))
        self.f.installAlgorithm('tg_label_prop')
        for v in v_type_set:
            self._create_vertex_attr(v, attr)
        
        try:
            result = self.f.runAlgorithm('tg_label_prop', params=label_propagation_params)
        except Exception as e:
            log.exception('failed to run label_propagation with params={}'.format(label_propagation_params, e))
            raise e
        
        log.info('label_propagation exit for params={}. Output: {}'.format(label_propagation_params, result))
        return result

    def kmeans(self, v_type: str, e_type: str, k=2, max_k=5, max_change=1.0):
        # TODO: send kmeans query to tigergraph, get list of similar users.
        kmeans_params = {
            "v_type": v_type,
            "e_type": e_type,
            "k": k,
            "max_k": max_k,
            "max_change": max_change
        }
        
        # build_embedding_gsql = """
        # USE GRAPH {}

        # CREATE QUERY build_embedding() FOR GRAPH {} {{
        #     start = {user.*};
        #     user_info = SELECT tgt
        #         FROM start:tgt
        #         POST_ACCUM
        #             tgt.kmeans_embedding = [tgt.pagerank_score, tgt.louvain_score, tgt.label_prop_score, tgt.degree_score];
        # }}

        # INSTALL QUERY build_embedding
        # """.format(self.conn.graphname, self.conn.graphname)


        with open(GSQL_PATH_PREFIX + 'tg_kmeans.gsql') as f, open(GSQL_PATH_PREFIX + 'tg_kmeans_sub.gsql') as g:
            tg_kmeans, tg_kmeans_sub = f.read().format(self.conn.graphname), g.read().format(self.conn.graphname)
        

        # result = self.conn.gsql(build_embedding_gsql)
        # return self.conn.runInstalledQuery('build_embedding')


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
    
    def _create_vertex_attr(self, v_type: str, attr: str):
        """Helper to generate and run GSQL to add new attribute to a vertex in graph.
        """
        try:
            schema_change_gsql = """
                CREATE SCHEMA_CHANGE JOB add_vertex_attribute FOR GRAPH {} {{
                    ALTER VERTEX {} ADD ATTRIBUTE ({} INT);
                }}
                RUN SCHEMA_CHANGE JOB add_vertex_attribute
            """.format(self.conn.graphname, v_type, attr)
            log.info('adding {} attribute to vertex {} with query {}'.format(attr, v_type, schema_change_gsql))
            self.conn.gsql(schema_change_gsql)
        except Exception as e:
            log.exception('failed to add {} attribute to vertex {}: {}'.format(attr, v_type, e))
            raise e