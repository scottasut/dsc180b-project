import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch import Tensor
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import tqdm
from sklearn.neighbors import NearestNeighbors
from src.util.tigergraph_util import get_subreddits
from src.models.baselines import PopularRecommender

class NetStatKNN:
    def __init__(self, conn):
        self.conn = conn

    def fit(self, user_data: pd.DataFrame, subreddit_data: pd.DataFrame, subreddit_interactions: pd.DataFrame, k: int):
        user_data.columns = list(range(len(user_data.columns)))
        users = user_data[0]
        self._knn_user_idx_map = {i:u for i, u in zip(range(len(users)), users)}
        self._user_knn_embeddings = user_data
        self._user_knn_embeddings.index = users
        self._user_knn_embeddings = self._user_knn_embeddings.drop(columns=[0])
        self._user_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k+1, n_jobs=-1).fit(self._user_knn_embeddings)
        self.subreddit_data = subreddit_data
        self._pop_rec = PopularRecommender(subreddit_interactions) 

    def recommend(self, user: str, n: int):
        if not self._user_knn:
            raise Exception('Must fit first.')
        _, indices = self._user_knn.kneighbors(np.array([self._user_knn_embeddings.loc[user]]))
        eligible_subs = set()
        user_subreddits = get_subreddits(self.conn, user)
        for i in indices[:2]:
            for j in i[1:]:
                other = self._knn_user_idx_map[j]
                other_subs = get_subreddits(self.conn, other)
                for sr in other_subs:
                    eligible_subs.add(sr)
        recommendations = list(eligible_subs - user_subreddits)
        subreddit_data = self.subreddit_data.loc[self.subreddit_data[0].isin(list(set(recommendations).union(user_subreddits)))]
        subreddit_data.columns = list(range(len(subreddit_data.columns)))
        subreddits = subreddit_data[0]
        knn_subreddit_idx_map = {i:u for i, u in zip(range(len(subreddits)), subreddits)}
        subreddit_knn_embeddings = subreddit_data
        subreddit_knn_embeddings.index = subreddits
        subreddit_knn_embeddings = subreddit_knn_embeddings.drop(columns=[0])
        subreddit_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1).fit(subreddit_knn_embeddings)
        _, indices = subreddit_knn.kneighbors(subreddit_knn_embeddings.loc[list(user_subreddits)])
        indices = indices.T

        recommendations = []
        recommended = set()
        for i in indices:
            if len(recommendations) == n:
                break
            for j in i:
                if len(recommendations) == n:
                    break
                subreddit = knn_subreddit_idx_map[j]
                if subreddit not in recommended and subreddit not in user_subreddits:
                    recommendations.append(subreddit)
                recommended.add(subreddit)
        
        while len(recommendations) < n:
            pop_recs = self._pop_rec.recommend(user, n=n - len(recommendations), avoid=recommendations)
            for pr in pop_recs:    
                recommendations.append(pr)
        return recommendations

class GNNHandler:
    def __init__(self, data, maps):
        self.data = data
        self.user_map, self.rev_user_map, self.subreddit_map, self.rev_subreddit_map = maps
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=('user', 'commented_in', 'subreddit'),
            rev_edge_types=('subreddit', 'rev_commented_in', 'user'), 
        )
        self.train_data, self.val_data, self.test_data = transform(data)
        train_edge_label_index = self.train_data['user', 'commented_in', 'subreddit'].edge_label_index
        train_edge_label = self.train_data['user', 'commented_in', 'subreddit'].edge_label
        self.train_loader = LinkNeighborLoader(
            data=self.train_data,
            num_neighbors=[20, 10],
            neg_sampling_ratio=2.0,
            edge_label_index=(('user', 'commented_in', 'subreddit'), train_edge_label_index),
            edge_label=train_edge_label,
            batch_size=128,
            shuffle=True,
        )

        test_edge_label_index = self.test_data['user', 'commented_in', 'subreddit'].edge_label_index
        test_edge_label = self.test_data['user', 'commented_in', 'subreddit'].edge_label
        self.test_loader = LinkNeighborLoader(
            data=self.test_data,
            num_neighbors=[20, 10],
            neg_sampling_ratio=2.0,
            edge_label_index=(('user', 'commented_in', 'subreddit'), test_edge_label_index),
            edge_label=test_edge_label,
            batch_size=128,
            shuffle=True,
        )
        self.model = None
        self.preds = None
        self.truth = None
    
    def set_model(self, hidden_channels):
        self.model = Model(self.train_data, hidden_channels=hidden_channels)

    def train(self, epochs):
        if not self.model:
            raise Exception('Must call \'set_model\' to define model first.')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss, total_examples = 0, 0
            for sample in tqdm.tqdm(self.train_loader):
                sample = sample.to(device)
                optimizer.zero_grad()
                prediction = self.model(sample)
                true_label = sample['user', 'commented_in', 'subreddit'].edge_label
                loss = F.binary_cross_entropy_with_logits(prediction, true_label)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * prediction.numel()
                total_examples += prediction.numel()
            print('Epoch {}, Loss: {}'.format(epoch, round(total_loss / total_examples, 4)))
    
    def predict(self):
        self.preds = []
        self.truth = []
        for sample in tqdm.tqdm(self.test_loader):
            self.preds.append(self.model(sample))
            self.truth.append(sample['user', 'commented_in', 'subreddit'].edge_label)

    def recommend(self, user, n=1):
        if not self.preds or not self.truth:
            raise Exception('Must call \'predict\' first.')

        user_node_idx = self.user_map[user]

        subreddit_node_idxs = self.train_data['subreddit'].node_id
        user_subreddit_edge_idxs = self.test_data['user', 'commented_in', 'subreddit'].edge_index

        user_subreddit_edge_mask = (user_subreddit_edge_idxs[0] == user_node_idx)
        subreddit_node_mask = np.isin(user_subreddit_edge_idxs[1], user_subreddit_edge_idxs)
        user_subreddit_mask = user_subreddit_edge_mask & subreddit_node_mask

        user_pred_scores = torch.cat(self.preds, dim=0)[user_subreddit_mask[:-3]]

        top_k_idxs = (-user_pred_scores).argsort()[:n]
        top_k_subreddit_nodes = subreddit_node_idxs[top_k_idxs]
        return [self.rev_subreddit_map[p.item()] for p in top_k_subreddit_nodes]

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_subreddit: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_subreddit = x_subreddit[edge_label_index[1]]
        return (edge_feat_user * edge_feat_subreddit).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        self.subreddit_lin = torch.nn.Linear(1250, hidden_channels, dtype=torch.float64)
        self.user_lin = torch.nn.Linear(1250, hidden_channels, dtype=torch.float64)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.subreddit_emb = torch.nn.Embedding(data["subreddit"].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_lin(data["user"].x).float() + self.user_emb(data["user"].node_id).float(),
          "subreddit": self.subreddit_lin(data["subreddit"].x).float() + self.subreddit_emb(data["subreddit"].node_id).float(),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["subreddit"],
            data["user", "commented_in", "subreddit"].edge_label_index,
        )
        return pred