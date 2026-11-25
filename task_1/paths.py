"""Infrastructure paths for Task 1."""
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
    "visualisations_dir": "visualisations"
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
        base = self.config.parent if key in self.settings_from_file else TASK_ROOT
        return (base / self.settings[key]).resolve()

    def input(self, key, data=None):
        return ((Path(data) if data is not None else self.location("data_dir"))
                / self.settings[f"{key}_file"]).resolve()

    def family_result(self, family, root=None):
        root = Path(root) if root is not None else self.location("family_dir")
        return root / family / "resultats" / self.families[family]

SCRIPTS = {family: ("Tresultar.py" if family == "T" else f"{family}resultat.py")
           for family in DEFAULTS["families"]}


def family_script(family):
    return TASK_ROOT / "family" / family / "code" / SCRIPTS[family]


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


def ridge_csv(family):
    suffix = {"T": "temporal", "F": "spectral"}.get(family, family.lower())
    return f"reve_{suffix}_ridge_results.csv"
