import argparse
import json
import zipfile
from pathlib import Path
from runpy import run_path

if __package__:
    from task_1.paths import Paths, prepare_output
else:
    # Direct invocation places only this family's code directory on sys.path.
    infrastructure = run_path(str(Path(__file__).resolve().parents[3] / "paths.py"))
    Paths = infrastructure["Paths"]
    prepare_output = infrastructure["prepare_output"]

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


NAMES = [
    "R001_AbsCorrelationMean",
    "R002_AbsCorrelationStd",
    "R003_EigenEntropy",
    "R004_ParticipationRatio",
]
NAMES += [
    f"R{5+i:03d}_Coherence_{band}"
    for i, band in enumerate(["Delta", "Theta", "Alpha", "Beta", "Gamma"])
]
NAMES += [
    f"R{10+i:03d}_PLI_{band}"
    for i, band in enumerate(["Delta", "Theta", "Alpha", "Beta", "Gamma"])
]

BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]
STATS = ["mean", "std", "median", "q75", "top10mean"]


def pair_statistics(values):
    if not np.isfinite(values).all():
        return np.full(5, np.nan)

    top = max(1, int(np.ceil(0.1 * len(values))))

    return np.array([
        values.mean(),
        values.std(ddof=0),
        np.median(values),
        np.quantile(values, 0.75),
        np.sort(values)[-top:].mean(),
    ])


def extract_epoch(
    x, fs, lowpass, amplitude_eps, eigen_eps, pair_batch
):
    channels, time = x.shape

    # Paires uniques i<j, sans diagonale.
    ii, jj = np.triu_indices(channels, k=1)

    pair_values = np.full((len(ii), 10), np.nan)
    global_values = np.full(14, np.nan)

    qc = {
        "nonfinite_channels": int(
            (~np.isfinite(x).all(axis=1)).sum()
        )
    }

    if not np.isfinite(x).all():
        return global_values, pair_values, np.full(54, np.nan), qc

    # R001-R004 : corrélation population.
    centered = x - x.mean(axis=1, keepdims=True)
    std = centered.std(axis=1, ddof=0)

    qc["constant_or_weak_channels"] = int(
        (std <= amplitude_eps).sum()
    )

    if np.all(std > amplitude_eps):
        z = centered / std[:, None]
        corr = np.clip((z @ z.T) / time, -1, 1)
        corr = (corr + corr.T) / 2

        absolute = np.abs(corr[ii, jj])
        global_values[:2] = [
            absolute.mean(),
            absolute.std(ddof=0),
        ]

        # Valeurs propres de la matrice SIGNÉE.
        eigen = np.maximum(
            np.linalg.eigvalsh(corr),
            eigen_eps,
        )
        eigen /= eigen.sum()

        global_values[2] = (
            -np.sum(eigen * np.log(eigen)) / np.log(channels)
        )
        global_values[3] = (
            eigen.sum() ** 2 / np.sum(eigen ** 2)
        )

    spectrum = np.fft.rfft(x, axis=-1)
    frequencies = np.fft.rfftfreq(time, 1 / fs)
    cap = min(45.0, lowpass, fs / 2)

    for band, (name, low, high) in enumerate(BANDS):
        mask = (
            (frequencies >= low)
            & (frequencies < min(high, cap))
        )

        if not mask.any():
            raise ValueError(
                f"Bande {name} sans bin FFT : "
                "vérifier fs, lowpass et durée."
            )

        # Cohérence en magnitude : moyenne des cross-spectres.
        band_spectrum = spectrum[:, mask]
        cross = (
            band_spectrum @ band_spectrum.conj().T
        ) / mask.sum()

        power = np.maximum(np.real(np.diag(cross)), 0)
        denominator = np.sqrt(power[ii] * power[jj])

        coherence = np.full(len(ii), np.nan)

        np.divide(
            np.abs(cross[ii, jj]),
            denominator,
            out=coherence,
            where=denominator > 0,
        )

        # PLI après masque FFT rectangulaire et Hilbert.
        filtered = np.fft.irfft(
            spectrum * mask,
            n=time,
            axis=-1,
        )
        analytic = hilbert(filtered, axis=-1)

        usable = (
            np.abs(analytic).mean(axis=-1) > amplitude_eps
        )
        good_pairs = usable[ii] & usable[jj]

        coherence[~good_pairs] = np.nan
        coherence = np.clip(coherence, 0, 1)

        phase = np.angle(analytic)
        pli = np.full(len(ii), np.nan)

        # Petits lots de paires pour limiter la mémoire.
        for start in range(0, len(ii), pair_batch):
            stop = min(start + pair_batch, len(ii))
            delta = (
                phase[ii[start:stop]]
                - phase[jj[start:stop]]
            )

            pli[start:stop] = np.abs(
                np.mean(np.sign(np.sin(delta)), axis=-1)
            )

        pli[~good_pairs] = np.nan

        pair_values[:, band] = coherence
        pair_values[:, 5 + band] = pli

        global_values[4 + band] = np.mean(coherence)
        global_values[9 + band] = np.mean(pli)

        qc[name + "_undefined_coherence_pairs"] = int(
            (~np.isfinite(coherence)).sum()
        )
        qc[name + "_undefined_pli_pairs"] = int(
            (~np.isfinite(pli)).sum()
        )

    # Quatre scalaires + dix blocs de cinq statistiques.
    expanded = np.concatenate(
        [global_values[:4]]
        + [
            pair_statistics(pair_values[:, j])
            for j in range(10)
        ]
    )

    return global_values, pair_values, expanded, qc


def header(handle):
    version = np.lib.format.read_magic(handle)

    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }

    if version not in readers:
        raise ValueError(
            f"Format NPY non pris en charge : {version}"
        )

    return readers[version](handle)


def info(path, key):
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + ".npy") as handle:
            return header(handle)


def choose_key(path, explicit, kind):
    with zipfile.ZipFile(path) as archive:
        keys = [
            key[:-4]
            for key in archive.namelist()
            if key.endswith(".npy")
        ]

    if explicit:
        if explicit not in keys:
            raise ValueError(
                f"Clé absente : {explicit}; clés={keys}"
            )
        return explicit

    if kind == "embedding" and "embeddings" in keys:
        return "embeddings"

    candidates = []

    for key in keys:
        shape, _, dtype = info(path, key)

        if dtype.kind not in "iuf":
            continue

        if (
            kind == "eeg"
            and key.lower() in (
                "x", "segments", "eeg", "signals", "data"
            )
            and len(shape) == 3
        ):
            candidates.append(key)

        elif kind == "embedding" and len(shape) >= 2:
            candidates.append(key)

    if len(candidates) != 1:
        raise ValueError(
            f"Clé {kind} ambiguë : {candidates}. "
            "Préciser --signal-key ou --embedding-key."
        )

    return candidates[0]


def batches(path, key, size):
    """Lire le tableau compressé par lots."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + ".npy") as handle:
            shape, fortran, dtype = header(handle)

            if fortran or dtype.kind not in "iuf" or not shape:
                raise ValueError(
                    "Tableau numérique C-order requis."
                )

            row_bytes = (
                int(np.prod(shape[1:])) * dtype.itemsize
            )

            for start in range(0, shape[0], size):
                count = min(size, shape[0] - start)
                raw = handle.read(count * row_bytes)

                if len(raw) != count * row_bytes:
                    raise ValueError("Tableau NPZ tronqué.")

                yield start, np.frombuffer(
                    raw, dtype=dtype
                ).reshape((count,) + shape[1:])


def labels_from(path):
    # N'utiliser que vos propres fichiers NPZ de confiance.
    with np.load(path, allow_pickle=True) as data:
        keys = [
            key for key in data.files
            if key.lower() in (
                "y", "labels", "label", "targets"
            )
        ]

        if len(keys) > 1:
            raise ValueError("Plusieurs clés de labels.")

        if not keys:
            return None

        values = np.asarray(data[keys[0]])

    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]

    if values.ndim != 1:
        raise ValueError("Labels attendus : vecteur N.")

    return values


def raw_r2(y, prediction):
    result = np.full(y.shape[1], np.nan)

    if len(y) >= 2:
        denominator = (
            (y - y.mean(axis=0)) ** 2
        ).sum(axis=0)

        good = (
            (np.ptp(y, axis=0) > 0)
            & (denominator > 0)
        )

        result[good] = 1 - (
            ((y[:, good] - prediction[:, good]) ** 2)
            .sum(axis=0)
            / denominator[good]
        )

    return result


def average(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    return (
        float(values[valid].mean())
        if valid.any() else np.nan
    )


def predict(model, x):
    return np.asarray(model.predict(x)).reshape(len(x), -1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--segments", default=None
    )
    parser.add_argument(
        "--embeddings", default=None
    )
    parser.add_argument("--signal-key")
    parser.add_argument("--embedding-key")
    parser.add_argument("--fs", type=float, default=200)
    parser.add_argument("--lowpass", type=float, default=45)
    parser.add_argument(
        "--amplitude-eps", type=float, default=1e-12
    )
    parser.add_argument(
        "--eigen-eps", type=float, default=1e-12
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--pair-batch-size", type=int, default=64
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument(
        "--targets",
        choices=["global", "summary"],
        default="global",
    )
    parser.add_argument("--out", default=None)

    args = parser.parse_args()
    paths = Paths(vars(args).pop('config'))
    args.segments = args.segments or str(paths.input("segments"))
    args.embeddings = args.embeddings or str(paths.input("embeddings"))
    args.out = args.out or str(paths.family_result("R"))

    if (
        args.batch_size < 1
        or args.pair_batch_size < 1
        or not np.isfinite(args.fs)
        or args.fs <= 0
        or not np.isfinite(args.lowpass)
        or args.lowpass <= 30
        or not np.isfinite(args.amplitude_eps)
        or args.amplitude_eps < 0
        or not np.isfinite(args.eigen_eps)
        or args.eigen_eps <= 0
        or any(
            not np.isfinite(value) or value <= 0
            for value in args.alphas
        )
    ):
        parser.error("Paramètres numériques invalides.")

    out = Path(args.out)
    prepare_output(out)

    if any(out.iterdir()):
        raise FileExistsError(
            f"{out} non vide : choisir un nouveau --out."
        )

    print("Lecture des en-têtes NPZ...", flush=True)

    signal_key = choose_key(
        args.segments, args.signal_key, "eeg"
    )
    embedding_key = choose_key(
        args.embeddings, args.embedding_key, "embedding"
    )

    shape, _, _ = info(args.segments, signal_key)
    embedding_shape, _, _ = info(
        args.embeddings, embedding_key
    )

    if (
        len(shape) != 3
        or shape[0] < 1
        or shape[1] < 2
        or shape[2] < 3
    ):
        raise ValueError(
            "EEG attendu N×C×T, avec au moins deux canaux."
        )

    n, channels, time = shape

    if (
        len(embedding_shape) < 2
        or embedding_shape[0] != n
        or min(embedding_shape[1:]) < 1
    ):
        raise ValueError("Embeddings incompatibles.")

    labels = labels_from(args.segments)
    embedding_labels = labels_from(args.embeddings)

    if any(
        values is not None and len(values) != n
        for values in (labels, embedding_labels)
    ):
        raise ValueError("Nombre de labels incompatible.")

    if (
        labels is not None
        and embedding_labels is not None
        and not np.array_equal(labels, embedding_labels)
    ):
        raise ValueError("Labels EEG/embeddings non alignés.")

    ii, jj = np.triu_indices(channels, k=1)
    pairs = np.column_stack([ii, jj])

    global_values = np.empty((n, 14))
    pair_values = np.empty((n, len(ii), 10))
    expanded = np.empty((n, 54))

    expanded_names = (
        NAMES[:4]
        + [
            name + ":" + stat
            for name in NAMES[4:]
            for stat in STATS
        ]
    )

    qc = []

    print(
        f"Extraction R : {n} essais, "
        f"{channels} canaux, {len(ii)} paires",
        flush=True,
    )

    for start, batch in batches(
        args.segments, signal_key, args.batch_size
    ):
        batch = np.asarray(batch, dtype=np.float64)

        for i, epoch in enumerate(batch):
            values, pair_epoch, expansion, diagnostic = extract_epoch(
                epoch,
                args.fs,
                args.lowpass,
                args.amplitude_eps,
                args.eigen_eps,
                args.pair_batch_size,
            )

            global_values[start + i] = values
            pair_values[start + i] = pair_epoch
            expanded[start + i] = expansion

            qc.append({
                "row_index": start + i,
                **diagnostic,
            })

        print(
            f"EEG : {start + len(batch)}/{n}",
            end="\r", flush=True,
        )

    print(flush=True)
    pd.DataFrame(qc).to_csv(out / "r_qc.csv", index=False)

    np.savez_compressed(
        out / "r_features.npz",
        global_features=global_values,
        feature_names=np.asarray(NAMES),
        pair_features=pair_values,
        pair_feature_names=np.asarray(NAMES[4:]),
        pair_indices=pairs,
        expanded_features=expanded,
        expanded_feature_names=np.asarray(expanded_names),
        row_index=np.arange(n),
        labels=labels if labels is not None else np.array([]),
        sampling_frequency=args.fs,
    )

    global_df = pd.DataFrame(global_values, columns=NAMES)
    global_df.insert(0, "row_index", np.arange(n))

    if labels is not None:
        global_df.insert(1, "label", labels)

    global_df.to_csv(out / "r_features.csv", index=False)

    pd.DataFrame(
        expanded, columns=expanded_names
    ).assign(
        row_index=np.arange(n)
    ).to_csv(
        out / "r_expanded_features.csv", index=False
    )

    embeddings = np.empty((n, embedding_shape[-1]))

    print("Lecture et pooling des embeddings...", flush=True)

    for start, batch in batches(
        args.embeddings, embedding_key, args.batch_size
    ):
        if batch.ndim > 2:
            pooled = batch.mean(
                axis=tuple(range(1, batch.ndim - 1)),
                dtype=np.float64,
            )
        else:
            pooled = batch

        embeddings[start:start + len(batch)] = pooled

        print(
            f"Embeddings : {start + len(batch)}/{n}",
            end="\r", flush=True,
        )

    print(flush=True)

    targets = (
        global_values
        if args.targets == "global"
        else expanded
    )

    valid = (
        np.isfinite(embeddings).all(axis=1)
        & np.isfinite(targets).all(axis=1)
    )
    original_rows = np.flatnonzero(valid)

    if len(original_rows) < 14:
        raise ValueError(
            "Moins de 14 essais finis : examiner r_qc.csv."
        )

    train, remaining = train_test_split(
        np.arange(len(original_rows)),
        test_size=0.30,
        random_state=args.seed,
    )
    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=args.seed,
    )

    split = np.full(n, "excluded_nonfinite", dtype=object)

    for name, indices in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        split[original_rows[indices]] = name

    pd.DataFrame({
        "row_index": np.arange(n),
        "split": split,
    }).to_csv(out / "splits.csv", index=False)

    e = embeddings[valid]
    targets = targets[valid]

    x = StandardScaler().fit(e[train]).transform(e)
    trainval = np.concatenate([train, validation])

    results = []
    dimension_results = []

    predictions = np.empty(
        (len(test), targets.shape[1])
    )

    print(
        f"Train={len(train)}, validation={len(validation)}, "
        f"test={len(test)}, exclus={int((~valid).sum())}",
        flush=True,
    )

    for j, name in enumerate(NAMES):
        if args.targets == "global" or j < 4:
            columns = slice(j, j + 1)
        else:
            start = 4 + 5 * (j - 4)
            columns = slice(start, start + 5)

        y = targets[:, columns]
        y_scaler = StandardScaler().fit(y[train])
        scaled_y = y_scaler.transform(y)

        scores = []

        for alpha in args.alphas:
            model = Ridge(alpha=alpha).fit(
                x[train], scaled_y[train]
            )

            scores.append(average(raw_r2(
                scaled_y[validation],
                predict(model, x[validation]),
            )))

        if np.isfinite(scores).any():
            best = int(np.argmax(np.where(
                np.isfinite(scores), scores, -np.inf
            )))
            status = "ok"
        else:
            best = int(np.argmax(args.alphas))
            status = "undefined_validation_r2_fallback_max_alpha"

        alpha = args.alphas[best]

        model = Ridge(alpha=alpha).fit(
            x[trainval], scaled_y[trainval]
        )
        prediction = y_scaler.inverse_transform(
            predict(model, x[test])
        )

        predictions[:, columns] = prediction
        r2 = raw_r2(y[test], prediction)

        results.append({
            "Feature": name,
            "Targets": args.targets,
            "Best_Alpha": alpha,
            "Validation_R2": scores[best],
            "Test_R2": average(r2),
            "Defined_Test_Dimensions": int(np.isfinite(r2).sum()),
            "Status": status,
        })

        dimension_names = (
            ["global"] if y.shape[1] == 1 else STATS
        )

        for k, dimension in enumerate(dimension_names):
            train_std = float(y[train, k].std())
            test_std = float(y[test, k].std())

            dimension_results.append({
                "Feature": name,
                "Dimension": dimension,
                "Test_R2": r2[k],
                "Test_RMSE": float(np.sqrt(np.mean(
                    (y[test, k] - prediction[:, k]) ** 2
                ))),
                "Test_MAE": float(np.mean(
                    np.abs(y[test, k] - prediction[:, k])
                )),
                "Train_Std": train_std,
                "TrainVal_Std": float(y[trainval, k].std()),
                "Test_Std": test_std,
                "Low_Variance_Train": bool(
                    train_std <=
                    1e-8 + 1e-6 * abs(y[train, k].mean())
                ),
                "Low_Variance_Test": bool(
                    test_std <=
                    1e-8 + 1e-6 * abs(y[test, k].mean())
                ),
                "Undefined_Test_R2": not np.isfinite(r2[k]),
            })

        print(
            f"{name:30s} alpha={alpha:g} "
            f"val R²={scores[best]:.4f} "
            f"test R²={average(r2):.4f}",
            flush=True,
        )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        out / "reve_r_ridge_results.csv", index=False
    )
    pd.DataFrame(dimension_results).to_csv(
        out / "reve_r_dimensions_results.csv", index=False
    )

    column_names = (
        NAMES if args.targets == "global" else expanded_names
    )

    np.savez_compressed(
        out / "test_predictions.npz",
        row_index=original_rows[test],
        y_true=targets[test],
        y_pred=predictions,
        column_names=np.asarray(column_names),
    )

    config = vars(args) | {
        "n_total": n,
        "n_valid": len(original_rows),
        "n_channels": channels,
        "n_pairs": len(ii),
        "source": "https://arxiv.org/pdf/2605.11410, eq.38-43 and C.3",
        "conventions": (
            "signed population Pearson correlation; "
            "clipped normalized eigenvalues; "
            "FFT-bin magnitude coherence, not squared/Welch; "
            "PLI abs(mean(sign(sin(delta_phase))))"
        ),
        "pair_summary": (
            "mean,std(ddof=0),median,q75,top ceil(10 percent) mean; "
            "unique i<j pairs"
        ),
        "undefined": (
            "weak/constant channels or weak band amplitudes -> NaN; "
            "incomplete target rows excluded"
        ),
        "split": "random trial-level 70/15/15; subjects may overlap",
        "scaling": "fit train; Ridge refit train+validation",
        "alignment": "original row order required; available labels checked",
        "r2": "raw; constants undefined; no clipping",
        "completed": True,
    }

    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== RÉSULTATS ==========")
    print(result_df.to_string(index=False))
    print(f"\nRésultats enregistrés dans : {out.resolve()}")


if __name__ == "__main__":
    main()
