from train import train
from pathlib import Path
from data.load_data import load_graph_excel

def test_training_smoke_org():
    """Load the small dummy Excel dataset and run 2 epochs to verify the training pipeline."""
    data_path = Path("data/smoke_data.xlsx")
    x, adj, y_true = load_graph_excel(data_path)
    final_loss = train(x, adj, y_true, epochs=2, option="org")
    assert final_loss >= 0.0

def test_training_smoke_dev():
    """Load the small dummy Excel dataset and run 2 epochs to verify the training pipeline."""
    data_path = Path("data/smoke_data.xlsx")
    x, adj, y_true = load_graph_excel(data_path)
    final_loss = train(x, adj, y_true, epochs=2, option="dev")
    assert final_loss >= 0.0