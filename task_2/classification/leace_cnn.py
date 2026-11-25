"""LEACE fitted on mean embeddings and applied to every token of a frozen CNN."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory, gettempdir
from zipfile import ZipFile

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import torch

from ..paths import FOLDS, Paths, output_problem, prepare_output
from .cnn_reve import (cache_embeddings, choose_device, load_classifier, load_labels,
                       load_metadata, mean_pool_embeddings, metrics, write_json)
from .leace_mlp import read_targets, retained


def fit_whitening(train):
    mean = train.mean(axis=0)
    centered = train - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    keep = retained(s, centered.shape)
    return mean, centered, u[:, keep], s[keep], vt[keep]


def fit_target(centered, u, s, vt, train_targets):
    target_mean = train_targets.mean(axis=0)
    z = train_targets - target_mean
    scale = z.std(axis=0)
    active = np.ptp(train_targets, axis=0) > 0
    scale = np.where(scale > 0, scale, 1)
    z[:, ~active] = 0
    z /= scale
    cross = u.T @ z
    if len(s):
        basis, sv, _ = np.linalg.svd(cross, full_matrices=False)
        basis = basis[:, retained(sv, cross.shape)]
    else:
        basis = np.empty((0, 0))
    left, right = projection_factors(s, vt, basis)
    clean_train = centered - (centered @ left) @ right
    residual = np.linalg.norm(clean_train.T @ z)
    fraction = residual / max(np.linalg.norm(cross * s[:, None]), np.finfo(float).tiny)
    if not np.isfinite(fraction) or fraction > 1e-7:
        raise ArithmeticError(f"Excessive relative training covariance after LEACE: {fraction}")
    return left, right, clean_train, fraction, target_mean, scale, active, z


def projection_factors(s, vt, basis):
    return (vt.T / s) @ basis, basis.T @ (s[:, None] * vt)


def random_factors(rng, s, vt, rank):
    # Same draw, reduced QR, whitening and recoloring as the MLP control.
    basis = np.linalg.qr(rng.normal(size=(len(s), rank)), mode="reduced")[0]
    return projection_factors(s, vt, basis)


def transform_tokens(tokens, mean, left, right):
    return tokens - ((tokens - mean) @ left) @ right


def infer(model, array, rows, batch_size, device, transform=None, saved_tokens=None):
    blocks = []
    squared = 0.
    count = 0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            source = np.asarray(array[rows[start:start + batch_size]], dtype=np.float32)
            block = source if transform is None else transform_tokens(source, *transform).astype(np.float32)
            if block.shape != source.shape or not np.isfinite(block).all():
                raise ValueError("Invalid dimensions or nonfinite CNN tokens")
            if transform is not None:
                change = source - block
                squared += np.einsum("ncpd,ncpd->", change, change, dtype=np.float64)
                count += change.size
            if saved_tokens is not None:
                saved_tokens[start:start + len(block)] = block
            blocks.append(model(torch.from_numpy(block).to(device)).softmax(dim=1).cpu().numpy())
    probabilities = np.concatenate(blocks)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite CNN probabilities")
    return probabilities, float(np.sqrt(squared / count)) if count else 0.


def probe_skill(train, test, z, target, active):
    # Same held-out comparison to a fixed training-mean predictor as LEACE MLP.
    if not active.any():
        return np.nan
    target = target[:, active]
    denominator = np.sum(target**2)
    if denominator == 0:
        return np.nan
    probe = Ridge(alpha=1., fit_intercept=False, solver="cholesky").fit(train, z[:, active])
    return 1 - np.sum((target - probe.predict(test).reshape(target.shape))**2) / denominator


def erase_feature(model, array, pooled, train, test, values, whitening, rng, controls,
                  batch_size, device, saved_tokens=None):
    mean, centered, u, s, vt = whitening
    left, right, clean, residual, target_mean, scale, active, z = fit_target(
        centered, u, s, vt, values[train])
    rank = left.shape[1]
    x = pooled[test] - mean
    changed = (x @ left) @ right
    target = (values[test] - target_mean) / scale
    diagnostics = dict(Rank=rank, Train_cov_residual=residual,
                       Probe_skill_original=probe_skill(centered, x, z, target, active),
                       Probe_skill_erased=probe_skill(clean, x - changed, z, target, active))
    for repetition in range(controls + 1):
        a, b = random_factors(rng, s, vt, rank) if repetition else (left, right)
        probabilities, token_rms = infer(model, array, test, batch_size, device, (mean, a, b),
                                         saved_tokens if not repetition else None)
        pooled_rms = np.sqrt(np.mean(((x @ a) @ b)**2))
        yield probabilities, a, b, dict(diagnostics, Control=repetition,
                                        Test_RMS_change=pooled_rms, Test_token_RMS_change=token_rms)


def check_disk(temp, out, member_size, output_estimate):
    destination = out.resolve()
    parent = next(p for p in (destination, *destination.parents) if p.exists())
    same_disk = temp.stat().st_dev == parent.stat().st_dev
    if shutil.disk_usage(temp).free < member_size + (output_estimate if same_disk else 0):
        raise OSError(f"Insufficient free space for CNN LEACE cache and outputs on {temp}")
    if not same_disk and shutil.disk_usage(parent).free < output_estimate:
        raise OSError(f"Insufficient free space for CNN LEACE outputs on {parent}")


@contextmanager
def token_output(path, shape, save):
    if not save:
        yield None
        return
    array = open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    try:
        yield array
        array.flush()
    finally:
        array._mmap.close()


def validate_baselines(folds, array, labels, subjects, config, batch_size, device):
    cohort = set(subjects)
    if (len(cohort) != len(FOLDS) or cohort != set(config["included_subjects"])
            or {c["test_subject"] for _, _, c in folds} != cohort
            or config.get("exclude_subjects") or not config.get("completed") or config.get("partial_run")):
        raise ValueError("Use the completed all-subject CNN LOSO experiment")
    required = np.concatenate([np.asarray(c["test_rows"]) for _, _, c in folds])
    n = len(labels)
    if (required.dtype.kind not in "iu" or len(np.unique(required)) != len(required)
            or len(required) != config["n_rows_used"] or np.any(required < 0) or np.any(required >= n)):
        raise ValueError("Invalid CNN test rows")
    for checkpoint, model, c in folds:
        subject = c["test_subject"]
        test, train = np.asarray(c["test_rows"]), np.asarray(c["train_rows"])
        if (test.dtype.kind not in "iu" or train.dtype.kind not in "iu"
                or c["model_config"].get("model_type", "cnn") != "cnn"
                or list(model.input_shape) != list(array.shape[1:])
                or c["source_shape"] != list(array.shape)
                or c["metadata_sha256"] != config["metadata_sha256"]
                or c["classes"] != config["classes"]
                or c["embedding_key"] != config["embedding_key"] or c["label_key"] != config["label_key"]
                or c["subject_column"] != config["subject_column"] or c.get("excluded_subjects")
                or set(c["train_subjects"]) != cohort - {subject}
                or not np.array_equal(np.sort(train), np.sort(required[subjects[required] != subject]))
                or not np.all(subjects[test] == subject)
                or not np.array_equal(labels[test], c["test_labels"])):
            raise ValueError(f"Incompatible CNN checkpoint or LOSO split: {subject}")
        y = np.asarray([c["classes"].index(v) for v in labels[test]])
        base, _ = infer(model, array, test, batch_size, device)
        saved = pd.read_csv(checkpoint.parent / "predictions.csv", dtype={"Subject": str}).set_index("row_index")
        cols = [f"Probability_Class_{i}" for i in range(len(c["classes"]))]
        if (not saved.index.is_unique or set(saved.index) != set(test)
                or not (saved.loc[test, "Subject"] == subject).all()
                or not np.array_equal(saved.loc[test, "True_Label"].astype(str), labels[test].astype(str))
                or not np.allclose(base, saved.loc[test, cols].to_numpy(), atol=1e-5)):
            raise ValueError(f"Historical CNN probabilities or test rows could not be reproduced: {subject}")
        yield required, model, c, train, test, y, base


def run(args):
    problem = output_problem(args.out)
    if problem:
        raise FileExistsError(problem)
    config = json.loads((args.models / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "cnn":
        raise ValueError("CNN training configuration required")
    labels = load_labels(args.embeddings, config["label_key"])
    _, subjects = load_metadata(args.metadata, len(labels), config["subject_column"])
    digest = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    if (digest != config["metadata_sha256"] or hashlib.sha256(
            labels.dtype.str.encode() + labels.tobytes()).hexdigest() != config["labels_sha256"]):
        raise ValueError("Metadata or labels differ from CNN training")
    device = choose_device(args.device)
    checkpoints = sorted(args.models.glob("fold_*/checkpoint.pt"))
    if [p.parent.name for p in checkpoints] != list(FOLDS):
        raise ValueError("Expected exactly ten CNN fold checkpoints")
    folds = [(path, *load_classifier(path, device)) for path in checkpoints]
    with ZipFile(args.embeddings) as archive:
        info = archive.getinfo(config["embedding_key"] + ".npy")
    source_config = config["source_embeddings"]
    if info.CRC != source_config["member_crc"] or info.file_size != source_config["member_size"]:
        raise ValueError("Embedding member differs from CNN training")
    shape = config["input_shape"]
    # Include all full token saves (if requested) and an upper bound for factors
    # and diagnostics, allowing each feature a full latent-dimensional rank.
    output_estimate = (4 * int(np.prod(shape)) * 63 if args.save_embeddings else 0)
    output_estimate += len(FOLDS) * (63 * (16 * shape[-1]**2 + 4 * shape[0] * 3) + 8 * shape[-1]**2)
    check_disk(Path(gettempdir()), args.out, info.file_size, output_estimate)
    with TemporaryDirectory(prefix="reve_leace_cnn_") as cache:
        array, provenance = cache_embeddings(args.embeddings, config["embedding_key"], cache)
        try:
            if list(array.shape) != shape or len(array) != len(labels):
                raise ValueError("Invalid CNN input shape")
            baselines = list(validate_baselines(folds, array, labels, subjects, config, args.batch_size, device))
            required = baselines[0][0]
            targets = read_targets(args.family, subjects, required, args.paths)
            pooled = mean_pool_embeddings(array, required, args.batch_size)[:, 0, 0].astype(float)
            prepare_output(args.out)
            write_json(args.out / "config.json", dict(
                model_type="cnn", method="common_affine_projection_on_tokens", completed=False,
                seed=args.seed, controls=args.controls, batch_size=args.batch_size, device=str(device),
                metadata_sha256=digest, labels_sha256=config["labels_sha256"], embeddings=provenance,
                models=str(args.models.resolve()), features=list(targets), save_embeddings=args.save_embeddings))
            run_interventions(args, baselines, targets, array, pooled, labels, device)
            result_config = json.loads((args.out / "config.json").read_text(encoding="utf-8"))
            write_json(args.out / "config.json", result_config | {"completed": True})
        finally:
            array._mmap.close()


def run_interventions(args, baselines, targets, array, pooled, labels, device):
    records = []
    for index, (_, model, c, train, test, y, base) in enumerate(baselines):
        subject = c["test_subject"]
        if Path(subject).name != subject or subject in (".", ".."):
            raise ValueError("Invalid subject directory name")
        folder = args.out / subject
        folder.mkdir()
        mean, centered, u, s, vt = whitening = fit_whitening(pooled[train])
        rng = np.random.default_rng(args.seed + index)
        baseline = metrics(y, base)
        np.savez_compressed(folder / "fold.npz", mean=mean, train_rows=train, test_rows=test, s=s, vt=vt,
                            classes=c["classes"], labels=labels[test], original_probabilities=base)
        for feature, values in targets.items():
            if Path(feature).name != feature or feature in (".", "..", "fold"):
                raise ValueError("Invalid feature filename")
            with token_output(folder / (feature + "_tokens.npy"), (len(test), *array.shape[1:]),
                              args.save_embeddings) as saved:
                for probability, left, right, diagnostic in erase_feature(
                        model, array, pooled, train, test, values, whitening,
                        rng, args.controls, args.batch_size, device, saved):
                    if diagnostic["Control"] == 0:
                        np.savez_compressed(folder / (feature + ".npz"), left=left, right=right,
                                            erased_probabilities=probability)
                    record = dict(Subject=subject, Feature=feature, **diagnostic)
                    for metric, value in metrics(y, probability).items():
                        record[metric] = value
                        record["Original_" + metric] = baseline[metric]
                        record["Delta_" + metric] = value - baseline[metric]
                    records.append(record)
        pd.DataFrame(records).to_csv(args.out / "by_subject.csv", index=False)
        print(f"{subject}: interventions and controls completed", flush=True)
    frame = pd.DataFrame(records)
    columns = [name for name in frame if name.startswith("Delta_")]
    columns += ["Test_RMS_change", "Test_token_RMS_change"]
    actual = frame[frame.Control == 0].groupby("Feature")[columns].mean()
    random = frame[frame.Control > 0].groupby(["Feature", "Subject"])[columns].mean().groupby("Feature").mean()
    actual.join(random.add_prefix("Random_")).to_csv(args.out / "summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "embeddings", "metadata", "family", "models", "out"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--controls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="Save targeted tokens only for test rows; fold.npz maps their indices")
    args = parser.parse_args()
    if min(args.controls, args.batch_size, args.threads) < 1 or args.seed < 0:
        parser.error("Positive controls, batch size and threads, and a nonnegative seed required")
    args.paths = paths = Paths(args.config)
    args.embeddings = args.embeddings or paths.input("embeddings", args.data)
    args.metadata = args.metadata or paths.input("metadata", args.data)
    args.family = args.family or paths.location("family_dir")
    args.models = args.models or paths.location("cnn_all_results_dir")
    args.out = args.out or paths.location("leace_cnn_results_dir")
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
