import torch
import torch.nn as nn
from model import SimpleGNNRegressor
from data.load_data import load_graph_excel
from pathlib import Path

def train(
    node_features: torch.Tensor,
    adj_matrix: torch.Tensor,
    target_distance: torch.Tensor,
    epochs: int = 100,
) -> float:
    in_features = node_features.shape[-1]
    model = SimpleGNNRegressor(in_features=in_features, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = model(node_features, adj_matrix)
        loss = criterion(y_pred, target_distance)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch}/{epochs}] - MSE Loss: {loss.item():.4f}")

    print("Training finished!")
    return loss.item()


if __name__ == "__main__":
    data_path = Path("data/full_data.xlsx")
    x, adj, y_true = load_graph_excel(data_path)
    train(x, adj, y_true, epochs=100)