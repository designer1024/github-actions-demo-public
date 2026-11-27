from __future__ import annotations

from pathlib import Path
import random

from openpyxl import Workbook, load_workbook
import torch


DEFAULT_FULL_DATA_PATH = Path("data/full_data.xlsx")
DEFAULT_SMOKE_DATA_PATH = Path("data/smoke_data.xlsx")


def save_dummy_excel(
    output_path: str | Path,
    num_samples: int,
    num_nodes: int = 5,
    in_features: int = 4,
    seed: int = 42,
) -> Path:
    """Generate random graph regression data and save it to an Excel workbook."""
    rng = random.Random(seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()

    # Sheet 1: targets
    ws_targets = wb.active
    ws_targets.title = "targets"
    ws_targets.append(["sample_id", "target_distance"])

    # Sheet 2: node_features
    ws_features = wb.create_sheet("node_features")
    ws_features.append(["sample_id", "node_id", *[f"f{i}" for i in range(in_features)]])

    # Sheet 3: adj_matrix
    ws_adj = wb.create_sheet("adj_matrix")
    ws_adj.append(["sample_id", "row_node", *[f"c{j}" for j in range(num_nodes)]])

    for sample_id in range(num_samples):
        ws_targets.append([sample_id, round(rng.uniform(0.0, 10.0), 6)])

        for node_id in range(num_nodes):
            feats = [round(rng.gauss(0.0, 1.0), 6) for _ in range(in_features)]
            ws_features.append([sample_id, node_id, *feats])

        raw = [[rng.random() for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            sym_row = [round((raw[i][j] + raw[j][i]) / 2.0, 6) for j in range(num_nodes)]
            ws_adj.append([sample_id, i, *sym_row])

    wb.save(path)
    return path


def generate_full_dataset(output_path: str | Path = DEFAULT_FULL_DATA_PATH) -> Path:
    """Generate a larger dummy Excel dataset to simulate normal training data."""
    return save_dummy_excel(output_path=output_path, num_samples=500, seed=42)


def generate_smoke_dataset(output_path: str | Path = DEFAULT_SMOKE_DATA_PATH) -> Path:
    """Generate a tiny dummy Excel dataset for fast CI smoke testing."""
    return save_dummy_excel(output_path=output_path, num_samples=8, seed=123)


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


if __name__ == "__main__":
    full_path = generate_full_dataset()
    smoke_path = generate_smoke_dataset()
    print(f"Generated full dataset:  {full_path}")
    print(f"Generated smoke dataset: {smoke_path}")