"""Project paths shared by local Jupyter and Google Colab runs."""
from pathlib import Path


def project_paths(root):
    root = Path(root).expanduser().resolve()
    paths = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed" / "cnn_lstm",
        "models": root / "models" / "cnn_lstm",
        "outputs": root / "outputs" / "cnn_lstm",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
