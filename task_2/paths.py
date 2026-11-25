"""Infrastructure paths for Task 2."""
import json
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_ROOT.parent
DEFAULTS = {
    "data_dir": "../data",
    "segments_file": "segments_labeled.npz",
    "embeddings_file": "reve_embeddings.npz",
    "family_dir": "family",
    "families": {
        "T": "results_temporal",
        "F": "results_spectral",
        "TF": "results_tf",
        "C": "results_c",
        "R": "results_r",
        "X": "results_x"
    },
    "metadata_file": "segments_metadata.csv",
    "classification_dir": "classification",
    "mlp_results_dir": "classification/results_mlp_tous_sujets",
    "cnn_results_dir": "classification/results_cnn_sans_P02",
    "leace_results_dir": "classification/results_leace_mlp",
    "visualisations_dir": "classification/results_leace_mlp/visualisations",
    "cnn_all_results_dir": "classification/results_cnn_tous_sujets",
    "leace_cnn_results_dir": "classification/results_leace_cnn",
    "cnn_visualisations_dir": "classification/results_leace_cnn/visualisations",
    "comparison_dir": "classification/comparison_mlp_cnn"
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
        self.families = DEFAULTS["families"] | settings.get("families", {})

    def location(self, key):
        if key in self.settings_from_file:
            return (self.config.parent / self.settings[key]).resolve()
        if key == "visualisations_dir":
            return self.location("leace_results_dir") / "visualisations"
        if key == "cnn_visualisations_dir":
            return self.location("leace_cnn_results_dir") / "visualisations"
        if key.endswith("_results_dir") or key == "comparison_dir":
            return self.location("classification_dir") / Path(DEFAULTS[key]).name
        return (TASK_ROOT / DEFAULTS[key]).resolve()

    def input(self, key, data=None):
        return ((Path(data) if data is not None else self.location("data_dir"))
                / self.settings[f"{key}_file"]).resolve()

    def family_result(self, family, root=None):
        root = Path(root) if root is not None else self.location("family_dir")
        return root / family / "resultats" / self.families[family]

FOLDS = tuple(f"fold_{fold:02d}" for fold in range(10))


def output_problem(path, resume=False):
    if path.exists():
        if not path.is_dir():
            return f"Output is not a directory: {path}"
        if any(path.iterdir()) and not (resume and (path / "config.json").is_file()):
            return f"Output directory already contains results: {path}. Choose another --out."
    return None


def prepare_output(path):
    problem = output_problem(path)
    if problem:
        raise FileExistsError(problem)
    path.mkdir(parents=True, exist_ok=True)


def leace_target(paths, family, root):
    target = paths.family_result(family, root) / "test_predictions.npz"
    if target.resolve().is_relative_to((PROJECT_ROOT / "task_1").resolve()):
        raise ValueError(f"LEACE requires Task 2 targets: {target}")
    return target
