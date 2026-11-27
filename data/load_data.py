from openpyxl import load_workbook
import torch
from pathlib import Path

def load_graph_excel(
    excel_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load graph features, adjacency matrices, and target distances from Excel."""
    wb = load_workbook(excel_path, read_only=True, data_only=True)

    targets = [float(row[1]) for row in wb["targets"].iter_rows(min_row=2, values_only=True)]
    num_samples = len(targets)

    feat_rows = list(wb["node_features"].iter_rows(min_row=2, values_only=True))
    adj_rows = list(wb["adj_matrix"].iter_rows(min_row=2, values_only=True))
    wb.close()

    num_nodes = len(feat_rows) // num_samples
    in_features = len(feat_rows[0]) - 2

    node_features = torch.tensor(
        [row[2:] for row in feat_rows], dtype=torch.float32
    ).view(num_samples, num_nodes, in_features)

    adj_matrix = torch.tensor(
        [row[2:] for row in adj_rows], dtype=torch.float32
    ).view(num_samples, num_nodes, num_nodes)

    target_distance = torch.tensor(targets, dtype=torch.float32)
    return node_features, adj_matrix, target_distance
