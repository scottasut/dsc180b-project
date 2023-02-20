import pyTigerGraph as tg
import torch
import torch_geometric
import scipy.sparse as ssp
import numpy as np
from torch_geometric.utils import coalesce
from tqdm.auto import trange
import json
import matplotlib.pyplot as plt

class GCNHandler:
    def __init__(self, config_path):
        self.config_path = config_path
        self.conn = self.connection()
        self.f = self.conn.gds.featurizer()
        self.generate_secret()
        graph_loader = self.conn.gds.graphLoader(
            num_batches=1,
            v_extra_feats={
                'user': ['name'],
                'subreddit': ['name']
            },
            e_extra_feats={
                "interacted_with": ['is_train', 'is_test'],
                'commented_in': ['is_train', 'is_test']
            },
            output_format = 'PyG'
        )
        self.data = graph_loader.data
        self.num_users = self.data['user'].name.shape[0]
        self.num_subreddits = self.data['subreddit'].name.shape[0]
        self.num_nodes = self.num_users + self.num_subreddits
        # Get user-item bi-adjacency matrix for testing
        self.test_user_item = coalesce(self.data["commented_in"]["edge_index"][:, self.data["commented_in"].is_test])
        # Convert the bi-adjacency matrix to the adjacency matrix of a unipartite graph. 
        # Items are indexed after users. This step is needed because LightGCN only supports unipartite graph.
        self.test_edge_index = self.test_user_item.clone().detach()
        self.test_edge_index[1] = self.test_edge_index[1] + self.num_users
        # Get test users.
        self.test_users = self.test_user_item[0].unique()
        # Convert the bi-adjacency matrix to a sparse matrix for computations later.
        test_user_item_sparse = ssp.coo_matrix((np.ones(self.test_user_item.shape[1]),
                                                self.test_user_item), 
                                            shape=(self.num_users, self.num_subreddits)).tocsr()
        self.test_user_item_sparse = test_user_item_sparse[self.test_users]

        # Get user-item bi-adjacency matrix for training
        self.train_user_item = coalesce(self.data["commented_in"]["edge_index"][:, self.data["commented_in"].is_train])
        # Convert the bi-adjacency matrix to the adjacency matrix of a unipartite graph. 
        # Items are indexed after users. This step is needed because LightGCN only supports unipartite graph.
        self.train_edge_index = self.train_user_item.clone().detach()
        self.train_edge_index[1] = self.train_edge_index[1] + self.num_users
        # Make the adjacency matrix symmetric.
        self.train_edge_index = torch.cat((self.train_edge_index, self.train_edge_index[[1,0]]), dim=1)
        # Convert the user-item bi-adjacency matrix to a sparse matrix for easier computation later.
        self.user_item_history = ssp.coo_matrix((np.ones(self.train_user_item.shape[1]), self.train_user_item), 
                                        shape=(self.num_users, self.num_subreddits)).tocsr()
        self.user_item_history = self.user_item_history[self.test_users]
        self.user_item_history = torch.sparse_csr_tensor(self.user_item_history.indptr, 
                                                    self.user_item_history.indices, 
                                                    self.user_item_history.data, 
                                                    size=self.user_item_history.shape)
        self.model_set = False
    
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

    def set_model(self, hp):
        self.gcn = LightGCN(self.num_nodes, hp['embedding_dim'], hp['num_layers'], hp['dropout'])
        self.optimizer = torch.optim.Adam(self.gcn.parameters(), lr=hp["lr"], weight_decay=hp["l2_penalty"])
        self.hp = hp
        self.model_set = True

    def train(self, plot=True):

        if not self.model_set:
            raise Exception('Must set the model before training.')

        metrics = {}
        metrics["loss_lgcn_train"]=[] 
        metrics["loss_lgcn_val"]=[]
        metrics["recall_k_val"]=[]
        metrics["precision_k_val"]=[]

        for epoch in trange(12):
            self.gcn.train()
            # Negative sampling. Randomly permute items to get negative user 
            # item edges.
            neg_edges = self.train_user_item.clone().detach()
            neg_edges[1] = neg_edges[1][torch.randperm(neg_edges.shape[1])] + self.num_users
            # Forward pass. Encode + decode
            h = self.lightgcn.encode(self.train_edge_index)
            pos_scores = self.gcn.decode(h, self.train_edge_index[:, :self.train_user_item.shape[1]])
            neg_scores = self.gcn.decode(h, neg_edges)
            # Loss and backprop
            loss = self.gcn.bpr_loss(pos_scores, neg_scores)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Logging
            metrics["loss_lgcn_train"].append(loss.item())
            
            # Evaluate
            self.gcn.eval()
            neg_edges = self.test_edge_index.clone().detach()
            neg_edges[1]=neg_edges[1][torch.randperm(neg_edges.shape[1])]
            with torch.no_grad():
                # Loss
                pos_scores = self.gcn.decode(h, self.test_edge_index)
                neg_scores = self.gcn.decode(h, neg_edges) 
                loss = self.gcn.bpr_loss(pos_scores, neg_scores)
                metrics["loss_lgcn_val"].append(loss.item())
                # Recall at k
                # Get recommendations using the current model
                recs = self.gcn.recommend(embedding=h, 
                                        src_index=self.test_users, 
                                        dst_index=torch.arange(self.num_users, self.num_nodes), 
                                        k=self.hp["topk"], 
                                        history=self.user_item_history)
                # Convert the recommendations to a sparse matrix
                recs = ssp.csr_matrix((np.ones(recs.shape[0]*recs.shape[1]),
                                    recs.flatten() - self.num_users,
                                    np.arange(0, recs.shape[0] * recs.shape[1] + 1, self.hp["topk"])),
                                    shape=(len(self.test_users), self.num_items))
                metrics["recall_k_val"].append(
                    (recs.multiply(self.test_user_item_sparse).sum(axis=1) / self.test_user_item_sparse.sum(axis=1)).mean())
                metrics["precision_k_val"].append(
                    (recs.multiply(self.test_user_item_sparse).sum(axis=1) / self.hp["topk"]).mean())
        
        if plot:
            fig,ax = plt.subplots(3,1, figsize=(10,9))
            ax[0].plot(metrics["loss_lgcn_train"], label="Train");
            ax[0].plot(metrics["loss_lgcn_val"], label="Valid");
            ax[0].set_ylabel("Loss", fontsize=12)
            ax[0].legend();
            ax[1].plot(metrics["recall_k_val"], color="C1");
            ax[1].set_ylabel("Recall at {}".format(self.hp["topk"]), fontsize=12);
            ax[1].set_xlabel("Step", fontsize=12);
            ax[2].plot(metrics["precision_k_val"], color="C1");
            ax[2].set_ylabel("Precision at {}".format(self.hp["topk"]), fontsize=12);
            ax[2].set_xlabel("Step", fontsize=12);

    def split_vertices(self, train=.9, test=.1):
        splitter = self.conn.gds.vertexSplitter(is_train=train, is_test=test)
        splitter.run()

    def split_edges(self, train=.9, test=.1):
        splitter = self.conn.gds.edgeSplitter(is_train=train, is_test=test)
        splitter.run()


class LightGCN(torch_geometric.nn.models.LightGCN):           
    def encode(self, edge_index: torch.Adj, 
               nodes: torch.Tensor = None) -> torch.Tensor:
        if nodes:
            x = self.embedding(nodes)
        else:
            x = self.embedding.weight
        out = x * self.alpha[0]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]
        return out
    
    def decode(self, embedding: torch.Tensor, node_pairs: torch.Adj) -> torch.Tensor:
        out_src = embedding[node_pairs[0]]
        out_dst = embedding[node_pairs[1]]
        return (out_src * out_dst).sum(dim=-1)
    
    def forward(self, edge_index: torch.Adj, nodes: torch.Tensor = None, 
                node_pairs: torch.Adj = None) -> torch.Tensor:
        if node_pairs is None:
            node_pairs = edge_index
        embed = self.encode(edge_index, nodes)
        out = self.decode(embed, node_pairs)
        return out
    
    def bpr_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        log_prob = F.logsigmoid(pos_scores - neg_scores).mean()
        return -log_prob
    
    def recommend(self, embedding: torch.Tensor, src_index: torch.Tensor = None,
                  dst_index: torch.Tensor = None, k: int = 1, history: torch.Tensor = None) -> torch.Tensor:
        out_src = embedding if src_index is None else embedding[src_index] 
        out_dst = embedding if dst_index is None else embedding[dst_index]
        
        pred = out_src @ out_dst.t()
        
        if history is not None:
            pred = torch.F.softmax(pred, dim=-1) + history.neg()
        
        top_index = pred.topk(k, dim=-1).indices

        if dst_index is not None:  # Map local top-indices to original indices.
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index