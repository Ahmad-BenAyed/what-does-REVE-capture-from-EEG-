"""Infrastructure paths for Task 1."""
import json
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_ROOT.parent
DEFAULTS = {
    "data_dir": "../data",
    "embeddings_file": "reve_embeddings.npz",
    "features_file": "features_63.npz",
    "metadata_file": "segments_metadata.csv",
    "results_dir": "results_loso_norm_sujet",
    "visualisations_dir": "visualisations",
}


class Paths:
    def __init__(self, config=None):
        self.config = Path(config).resolve() if config else TASK_ROOT / "config.json"
        settings = {}
        if config is not None or self.config.exists():
            settings = json.loads(self.config.read_text(encoding="utf-8"))
            if not isinstance(settings, dict):
                raise ValueError(f"Expected a JSON object in {self.config}")
        self.settings_from_file = settings
        self.settings = DEFAULTS | settings

    def location(self, key):
        base = self.config.parent if key in self.settings_from_file else TASK_ROOT
        return (base / self.settings[key]).resolve()

    def input(self, key, data=None):
        return ((Path(data) if data is not None else self.location("data_dir"))
                / self.settings[f"{key}_file"]).resolve()


def output_problem(path):
    if path.exists():
        if not path.is_dir():
            return f"Output is not a directory: {path}"
        if any(path.iterdir()):
            return f"Output directory already contains results: {path}. Choose another --out."
    return None
