import torch
import torch.nn as nn

class SimpleGNNRegressor(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 16, option: str = "org"):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.regressor = nn.Linear(hidden_dim, 1)
        self.option = option

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        node_features: [batch_size, num_nodes, in_features]
        adj_matrix:    [batch_size, num_nodes, num_nodes]
        """
        agg = torch.bmm(adj_matrix, node_features)

        if self.option == "org":
            agg = self.fun(agg)
        elif self.option == "dev":
            agg = self.fun_dev(agg)
        else:
            raise ValueError(f"Invalid option: {self.option}")

        h = self.relu(self.linear1(agg))
        graph_emb = h.mean(dim=1)
        pred_distance = self.regressor(graph_emb)
        return pred_distance.squeeze(-1)

    def fun(self, x: torch.Tensor) -> torch.Tensor:
        print("function is called")
        return x

    def fun_dev(self, x: torch.Tensor) -> torch.Tensor:
        # Intentionally introduce a bug that doubles the feature dimension (4 -> 8), causing a shape mismatch error in the subsequent linear1(4, 16) layer
        print("fun_dev is called")
        return torch.cat([x, x], dim=-1)