import torch
import torch.nn as nn

class SimpleGNNRegressor(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        node_features: [batch_size, num_nodes, in_features]
        adj_matrix:    [batch_size, num_nodes, num_nodes]
        """
        agg = torch.bmm(adj_matrix, node_features)

        agg = self.fun(agg)

        h = self.relu(self.linear1(agg))
        graph_emb = h.mean(dim=1)
        pred_distance = self.regressor(graph_emb)
        return pred_distance.squeeze(-1)

    def fun(self, x: torch.Tensor) -> torch.Tensor:
        print("function is called")
        return x