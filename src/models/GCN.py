import pyTigerGraph as tg
import torch
import torch_geometric
import scipy.sparse as ssp
import numpy as np
from torch_geometric.utils import coalesce
from torch_geometric.typing import Adj
import torch.nn.functional as F
from torch_geometric.nn import GCN
from tqdm.auto import trange
import json
import matplotlib.pyplot as plt
import src.util.tigergraph_util as tgu

class GCNHandler:
    def __init__(self, config_path, num_batches):
        self.config_path = config_path
        self.conn = tgu.connection()
        self.f = self.conn.gds.featurizer()
        self.num_batches = num_batches
        self.graph_loader = tgu.load_graph(num_batches=num_batches)
        self.data = self.graph_loader.data
    
    def split_vertices(self, train=.9, test=.1):
        splitter = self.conn.gds.vertexSplitter(is_train=train, is_test=test)
        splitter.run()

    def split_edges(self, train=.9, test=.1):
        splitter = self.conn.gds.edgeSplitter(is_train=train, is_test=test)
        splitter.run()

    def set_model(self, hp):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GCN(
            in_channels=1433,
            hidden_channels=hp["hidden_dim"],
            num_layers=hp["num_layers"],
            out_channels=2,
            dropout=hp["dropout"],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=hp["lr"], weight_decay=hp["l2_penalty"]
        )

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

class LightGCN(torch_geometric.nn.models.LightGCN):           
    def encode(self, edge_index: Adj, 
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
    
    def decode(self, embedding: torch.Tensor, node_pairs: Adj) -> torch.Tensor:
        out_src = embedding[node_pairs[0]]
        out_dst = embedding[node_pairs[1]]
        return (out_src * out_dst).sum(dim=-1)
    
    def forward(self, edge_index: Adj, nodes: torch.Tensor = None, 
                node_pairs: Adj = None) -> torch.Tensor:
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