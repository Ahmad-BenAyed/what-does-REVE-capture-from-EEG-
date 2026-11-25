"""Check Task 2 files and headers without inflating large NPZ members."""
import csv
import hashlib
import json
from zipfile import BadZipFile, ZipFile, is_zipfile

if __package__:
    from .paths import FOLDS, leace_target, output_problem
else:
    from paths import FOLDS, leace_target, output_problem

import os
from importlib.util import find_spec


def check_output(path, resume=False):
    problem = output_problem(path, resume)
    if problem:
        print(f"[WARNING] {problem}")
        return False
    parent = next(folder for folder in (path, *path.parents) if folder.exists())
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        print(f"[ERROR] Output location is not writable: {path}")
        return False
    print(f"[OK] Output location: {path}")
    return True


def check_file(path):
    try:
        with path.open("rb"):
            pass
    except OSError as error:
        print(f"[ERROR] {path}: {error.strerror}")
        return False
    print(f"[OK] Readable file: {path}")
    return True


def check_dependencies(names):
    results = []
    for name in names:
        found = find_spec(name) is not None
        print(f"{'[OK]' if found else '[ERROR]'} Python dependency {name}: "
              f"{'available' if found else 'missing'}")
        results.append(found)
    return all(results)


DEPENDENCIES = {
    "train": ("numpy", "pandas", "torch", "sklearn"),
    "evaluate": ("numpy", "pandas", "torch", "sklearn"),
    "leace": ("numpy", "pandas", "torch", "sklearn", "threadpoolctl"),
    "visualize": ("numpy", "pandas", "matplotlib"),
    "compare": ("numpy", "pandas", "matplotlib"),
}


def npz_headers(path, *keys):
    from numpy.lib import format as npy

    headers = []
    with ZipFile(path) as archive:
        for key in keys:
            with archive.open(f"{key}.npy") as stream:
                version = npy.read_magic(stream)
                readers = {(1, 0): npy.read_array_header_1_0, (2, 0): npy.read_array_header_2_0}
                if version not in readers:
                    if version != (3, 0):
                        raise ValueError(f"Unsupported NPY version {version}")
                    # NumPy's public v2 reader uses Latin-1. v3 headers use UTF-8.
                    shape, _, dtype = npy._read_array_header(stream, version)
                else:
                    shape, _, dtype = readers[version](stream)
                if not shape or not all(shape) or dtype.hasobject:
                    raise ValueError(f"Invalid array header for {key} in {path}")
                headers.append((shape, dtype))
    return headers


def csv_info(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader, [])
        if not set(required).issubset(columns):
            raise ValueError(f"Missing CSV columns in {path}: {sorted(set(required) - set(columns))}")
        return sum(1 for row in reader if row), columns


def checkpoint_header(path):
    if is_zipfile(path):
        with ZipFile(path) as archive:
            if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                raise ValueError(f"Missing checkpoint payload: {path}")
    else:
        with path.open("rb") as stream:
            if not stream.read(1).startswith(b"\x80"):
                raise ValueError(f"Unrecognised checkpoint container: {path}")


def validate_data(args, stage):
    (shape, dtype), (labels, _) = npz_headers(args.embeddings, args.embedding_key, args.label_key)
    if len(shape) not in (2, 3, 4) or dtype.kind not in "iuf":
        raise ValueError("Embeddings must be a numeric array with 2–4 dimensions")
    n = shape[0]
    if labels not in ((n,), (n, 1)) or csv_info(args.metadata, (args.subject_column,))[0] != n:
        raise ValueError("Embeddings, labels and metadata have inconsistent row counts")
    if stage == "train":
        (segment_labels, _), = npz_headers(args.segments, args.label_key)
        if segment_labels not in ((n,), (n, 1)):
            raise ValueError("Segment labels and embeddings have inconsistent row counts")
    return shape


def validate_frozen_inputs(args):
    # The checkpoint defines the actual keys. Check for a plausible numeric
    # embedding member without deserializing the model during preflight.
    checkpoint_header(args.checkpoint)
    rows, _ = csv_info(args.metadata, ())
    if rows == 0:
        raise ValueError("Metadata has no rows")
    with ZipFile(args.embeddings) as archive:
        keys = [name[:-4] for name in archive.namelist() if name.endswith(".npy")]
    for key in keys:
        try:
            (shape, dtype), = npz_headers(args.embeddings, key)
        except (ValueError, EOFError):
            continue
        if len(shape) in (2, 3, 4) and dtype.kind in "iuf" and shape[0] == rows:
            return
    raise ValueError("No numeric embedding array matches the metadata row count")


def validate_leace(args, shape, targets, model_config=None):
    config = model_config or json.loads((args.models / "config.json").read_text(encoding="utf-8"))
    if model_config is not None and (config.get("model_type") != "cnn" or not config.get("completed")
                                     or config.get("partial_run") or config.get("exclude_subjects")):
        raise ValueError("Completed all-subject CNN configuration required")
    if not all(key in config for key in ("metadata_sha256", "labels_sha256", "source_embeddings", "input_shape")):
        raise ValueError("MLP config misses provenance fields")
    canonical = {2: lambda: [shape[0], 1, 1, shape[-1]],
                 3: lambda: [shape[0], shape[1], 1, shape[-1]],
                 4: lambda: list(shape)}[len(shape)]()
    if config["input_shape"] != canonical:
        raise ValueError("Embedding shape differs from the trained model configuration")
    with ZipFile(args.embeddings) as archive:
        if config["source_embeddings"]["member_crc"] != archive.getinfo(args.embedding_key + ".npy").CRC:
            raise ValueError("Embedding member CRC differs from the trained model configuration")
    for path in targets:
        (rows, dtype), (subjects, _), (names, _), (values, value_type) = npz_headers(
            path, "row_index", "subject", "column_names", "y_true"
        )
        if (len(rows) != 1 or dtype.kind not in "iu" or rows != subjects
                or rows[0] > shape[0] or len(names) != 1
                or values != (rows[0], names[0]) or value_type.kind not in "iuf"):
            raise ValueError(f"Inconsistent LEACE target headers: {path}")
    if {p.parent.name for p in args.models.glob("fold_*/checkpoint.pt")} != set(FOLDS):
        raise ValueError("Expected exactly the ten LOSO checkpoint directories")
    for fold in FOLDS:
        folder = args.models / fold
        checkpoint_header(folder / "checkpoint.pt")
        n, columns = csv_info(folder / "predictions.csv", ("row_index",))
        if n == 0 or not any(column.startswith("Probability_Class_") for column in columns):
            raise ValueError(f"Missing saved prediction probabilities: {folder}")


def deep_check(args, stage):
    # Only label members are inflated. The embeddings member stays compressed.
    import numpy as np

    with np.load(args.embeddings, allow_pickle=False) as archive:
        labels = archive[args.label_key].reshape(-1)
    if stage == "train":
        with np.load(args.segments, allow_pickle=False) as archive:
            if not np.array_equal(labels, archive[args.label_key].reshape(-1)):
                raise ValueError("Segment labels differ from embedding labels")
    else:
        config = json.loads((args.models / "config.json").read_text(encoding="utf-8"))
        meta_hash = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
        label_hash = hashlib.sha256(labels.dtype.str.encode() + labels.tobytes()).hexdigest()
        if meta_hash != config["metadata_sha256"] or label_hash != config["labels_sha256"]:
            raise ValueError("Metadata or labels differ from the saved MLP experiment")
    print("[OK] Deep label/provenance check")


def comparison_files(args):
    files = [folder / name for folder in (args.mlp, args.cnn)
             for name in ("config.json", "predictions.csv", "fold_metrics.csv")]
    if not args.classification_only:
        files.extend(folder / name for folder in (args.mlp_leace, args.cnn_leace)
                     for name in ("config.json", "by_subject.csv"))
    return files


def check(args, paths, stage):
    dependencies = check_dependencies(DEPENDENCIES[stage])
    targets = []
    if stage == "compare":
        files = comparison_files(args)
    elif stage == "visualize":
        files = [args.results / "by_subject.csv"]
    else:
        files = [args.embeddings, args.metadata]
        if stage == "train":
            files.append(args.segments)
        elif stage == "evaluate":
            files.append(args.checkpoint)
            if args.reference:
                files.append(args.reference)
        else:
            targets = [leace_target(paths, family, args.family)
                       for family in paths.families]
            files.extend([*targets, args.models / "config.json"])
            for fold in FOLDS:
                files.extend(args.models / fold / name for name in ("checkpoint.pt", "predictions.csv"))
    present = all([check_file(path) for path in files])
    writable = check_output(args.out, stage == "train" and args.resume)
    if not (dependencies and present):
        return False
    try:
        if stage == "compare":
            for folder in (args.mlp, args.cnn):
                csv_info(folder / "predictions.csv", ("row_index", "Subject", "True_Label", "Predicted_Label"))
                csv_info(folder / "fold_metrics.csv", ("Subject", "Accuracy", "Balanced_Accuracy", "Macro_F1", "Log_Loss"))
            if not args.classification_only:
                for folder in (args.mlp_leace, args.cnn_leace):
                    csv_info(folder / "by_subject.csv", ("Feature", "Subject", "Control", "Rank"))
                    if len(list(folder.glob("*/fold.npz"))) != len(FOLDS):
                        raise ValueError(f"Expected ten LEACE fold archives: {folder}")
        elif stage == "visualize":
            csv_info(files[0], ("Feature", "Subject", "Control"))
        elif stage == "evaluate":
            validate_frozen_inputs(args)
        else:
            model_config = None
            if args.step == "leace_cnn":
                model_config = json.loads((args.models / "config.json").read_text(encoding="utf-8"))
                args.embedding_key = model_config["embedding_key"]
                args.label_key = model_config["label_key"]
                args.subject_column = model_config["subject_column"]
            shape = validate_data(args, stage)
            if stage == "leace":
                validate_leace(args, shape, targets, model_config)
            if args.deep:
                deep_check(args, stage)
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile, csv.Error) as error:
        print(f"[ERROR] Input structure: {error}")
        return False
    print("[OK] Input structure")
    return writable


def main():
    if __package__:
        from .run_task_2 import main as run_main
    else:
        from run_task_2 import main as run_main
    import sys

    return run_main(["--check", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
