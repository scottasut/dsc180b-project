import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch.nn.functional as F
import torch_geometric.transforms as T
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
                if subreddit not in recommended:
                    recommendations.append(subreddit)
                recommended.add(subreddit)
        
        while len(recommendations) < n:
            pop_recs = self._pop_rec.recommend(user, n=n - len(recommendations), avoid=recommendations)
            for pr in pop_recs:    
                recommendations.append(pr)
        return recommendations

class LightGCNHandler:
    def __init__(self, user, subreddit, user_user, user_subreddit, test_val_split=(0.2, 0.1)):

        if sum(test_val_split) >= 1:
            raise ValueError('\'train_val_split\' must sum to less than 1.')
    
        self.data = HeteroData()
        self._user_idx_map = {u:i for i, u in enumerate(user[0])}
        self._user_idx_reverse_map = {i:u for u, i in self._user_idx_map.items()}
        self.data['user'].x = torch.tensor(user.drop(columns=[0]).values)
        self._subreddit_idx_map = {s:i for i, s in enumerate(subreddit[0])}
        self._subreddit_idx_reverse_map = {i:s for s, i in self._subreddit_idx_map.items()}
        self.data['subreddit'].x = torch.tensor(subreddit.drop(columns=[0]).values)
        user_user[0], user_user[1] = user_user[0].map(self._user_idx_map), user_user[1].map(self._user_idx_map)
        self.data['user', 'interacted_with', 'user'] = torch.tensor(user_user.values)
        user_subreddit[0], user_subreddit[1] = user_subreddit[0].map(self._user_idx_map), user_subreddit[1].map(self._subreddit_idx_map)
        self.data['user', 'commented_in', 'subreddit'] = torch.tensor(user_subreddit.values)
        
        self.train_test_split = test_val_split
        splitter = T.RandomNodeSplit(num_test=test_val_split[0], num_val=test_val_split[1])
        
    
    def set_model(self):
        pass

    def train_helper(self, epoch):
        self.model.train()
        epoch_train_loss = []
        epoch_train_acc = torch.Accuracy()
        for batch_num, data in enumerate(self.graph_loader):
            data.to(self.device)

            # Forward pass
            out = self.model(data.x, data.edge_index)

            # Calculate loss
            loss = F.cross_entropy(out[data.is_train], data.y[data.is_train])

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_train_loss.append(loss)

            # Predict on Training data
            with torch.no_grad():
                pred = out.argmax(dim=1)
                epoch_train_acc.update(pred[data.is_train], data.y[data.is_train])

            print('Epoch: {}, Training Batch: {}, Train accuracy: {:.4f}, Train loss: {:.4f}'.format(epoch, batch_num, epoch_train_acc.value, np.mean(epoch_train_loss)))


    def validate(self):
        self.model.eval()
        val_acc = torch.Accuracy()
        val_loss = []
        with torch.no_grad():
            for data in self.data_loader:
                # Forward pass
                out = self.model(data.x, data.edge_index)
                # Getting predictions
                pred = out.max(dim=1)
                # Validation loss
                loss = F.cross_entropy(out[data.is_test], data.y[data.is_test])
                val_loss.append(loss)
                # Validation accuracy
                val_acc.update(pred[data.is_test], data.y[data.is_test])

        return val_acc, np.mean(val_loss)

    def train(self):
        for epoch in range(self.num_batches):
            self.train_helper(epoch)
            acc, loss = self.validate()
            print('Epoch: {}, Val accuracy: {:.4f}, Val loss: {:.4f}'.format(epoch, acc, loss))
        

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, conv_model):
        super(GNNStack, self).__init__()
        conv_model = pyg.nn.SAGEConv


        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        self.dropout = dropout
        self.num_layers = num_layers


        # Create num_layers GraphSAGE convs
        assert (self.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(self.num_layers - 1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))


        # post-message-passing processing 
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_dim))


    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)


        x = self.post_mp(x)


        # Return final layer of embeddings if specified
        return x
