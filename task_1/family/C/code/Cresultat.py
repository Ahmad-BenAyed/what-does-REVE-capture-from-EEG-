import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


NAMES = [
    "C001_PermutationEntropy",
    "C002_SampleEntropyProxy",
    "C003_LZIrregularityProxy",
    "C004_HiguchiSlopeProxy",
    "C005_DFAProxy",
    "C006_ACFDecay",
    "C007_ACFFirstZero",
]


def entropy(counts, states):
    p = np.asarray(counts, dtype=float)
    p = p[p > 0] / p.sum()
    return -np.sum(p * np.log(p)) / np.log(states)


def slope(x, y):
    x, y = np.asarray(x), np.asarray(y)
    keep = np.isfinite(x) & np.isfinite(y)

    if keep.sum() < 2:
        return np.nan

    xx, yy = x[keep], y[keep]
    xx = xx - xx.mean()
    denominator = np.sum(xx * xx)

    if denominator <= 0:
        return np.nan

    return np.sum(xx * (yy - yy.mean())) / denominator


def extract_channel(x, max_lag):
    """Proxys de l'annexe A, équations 27–33."""
    if not np.isfinite(x).all():
        return np.full(7, np.nan), np.nan

    n = len(x)

    # C001 : entropie de permutation, ordre 3, lag 1.
    # Les égalités sont résolues par l'ordre temporel stable.
    windows = np.lib.stride_tricks.sliding_window_view(x, 3)
    patterns = np.argsort(windows, axis=-1, kind="stable")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    c1 = entropy(counts, 6)

    # C002 : proxy d'irrégularité des dérivées.
    # Différences brutes, sans multiplication par fs.
    dx = np.diff(x)
    ddx = np.diff(x, n=2)
    radius = 0.2 * np.std(x, ddof=0)

    if radius > 0:
        c2 = (
            np.log1p(np.mean(np.abs(ddx)) / radius)
            - np.log1p(np.mean(np.abs(dx)) / radius)
        )
    else:
        c2 = np.nan

    # C003 : transitions binaires + entropie des paires.
    binary = (x > np.median(x)).astype(np.int8)
    pairs = 2 * binary[:-1] + binary[1:]
    transitions = np.mean(binary[1:] != binary[:-1])

    c3 = (
        0.5 * transitions
        + 0.5 * entropy(np.bincount(pairs, minlength=4), 4)
    )

    # C004 : pente log des différences aux retards k.
    ks = np.arange(1, min(8, n // 4) + 1)

    lengths = np.array([
        np.mean(np.abs(x[k:] - x[:-k]))
        for k in ks
    ])

    positive = lengths > 0
    raw_c4 = slope(
        np.log(1 / ks[positive]),
        np.log(lengths[positive]),
    )
    c4 = np.clip(raw_c4, 0, 3)

    # C005 : DFA proxy, fenêtres complètes et detrending linéaire.
    integrated = np.cumsum(x - x.mean())
    scales = []
    fluctuations = []

    for size in (16, 32, 64, 128, 256, 512):
        if 2 * size > n:
            continue

        segments = integrated[:(n // size) * size].reshape(-1, size)
        time = np.arange(size, dtype=float)
        time -= time.mean()

        centered = segments - segments.mean(axis=1, keepdims=True)
        coefficient = (centered @ time) / np.sum(time ** 2)
        residual = centered - coefficient[:, None] * time
        fluctuation = np.sqrt(np.mean(residual ** 2))

        if fluctuation > 0:
            scales.append(size)
            fluctuations.append(fluctuation)

    c5 = np.clip(
        slope(np.log(scales), np.log(fluctuations)),
        -1, 2,
    )

    # C006/C007 : autocorrélation linéaire, centrée et normalisée.
    # Zero-padding pour éviter une autocorrélation circulaire.
    centered = x - x.mean()
    spectrum = np.fft.rfft(centered, n=2 * n)

    acf = np.fft.irfft(
        spectrum * spectrum.conj(),
        n=2 * n,
    )[:max_lag + 1]

    c6 = c7 = np.nan

    if acf[0] > 0:
        acf = acf / acf[0]
        decay = np.flatnonzero(acf[1:] <= 1 / np.e)
        zero = np.flatnonzero(acf[1:] <= 0)

        lag_decay = decay[0] + 1 if len(decay) else max_lag
        lag_zero = zero[0] + 1 if len(zero) else max_lag

        c6 = lag_decay / max_lag
        c7 = lag_zero / max_lag

    return np.array([c1, c2, c3, c4, c5, c6, c7]), raw_c4


def header(handle):
    version = np.lib.format.read_magic(handle)

    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }

    if version not in readers:
        raise ValueError(f"Version NPY non prise en charge : {version}")

    return readers[version](handle)


def info(path, key):
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + ".npy") as handle:
            return header(handle)


def choose_key(path, explicit, kind):
    with zipfile.ZipFile(path) as archive:
        keys = [
            name[:-4]
            for name in archive.namelist()
            if name.endswith(".npy")
        ]

    if explicit:
        if explicit not in keys:
            raise ValueError(f"Clé absente : {explicit}; clés={keys}")
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
            and key.lower() in ("x", "segments", "eeg", "signals", "data")
            and len(shape) in (2, 3)
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


def batches(path, key, batch_size):
    """Lecture NPZ par lots, sans chargement complet."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + ".npy") as handle:
            shape, fortran, dtype = header(handle)

            if fortran or dtype.kind not in "iuf" or not shape:
                raise ValueError("Tableau numérique C-order requis.")

            row_bytes = int(np.prod(shape[1:])) * dtype.itemsize

            for start in range(0, shape[0], batch_size):
                count = min(batch_size, shape[0] - start)
                raw = handle.read(count * row_bytes)

                if len(raw) != count * row_bytes:
                    raise ValueError("Tableau NPZ tronqué.")

                yield start, np.frombuffer(raw, dtype=dtype).reshape(
                    (count,) + shape[1:]
                )


def labels_from(path):
    # Utiliser uniquement vos propres fichiers NPZ de confiance.
    with np.load(path, allow_pickle=True) as data:
        keys = [
            key for key in data.files
            if key.lower() in ("y", "labels", "label", "targets")
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
        denominator = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        good = (np.ptp(y, axis=0) > 0) & (denominator > 0)

        result[good] = 1 - (
            ((y[:, good] - prediction[:, good]) ** 2).sum(axis=0)
            / denominator[good]
        )

    return result


def average(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    return float(values[valid].mean()) if valid.any() else np.nan


def predict(model, x):
    return np.asarray(model.predict(x)).reshape(len(x), -1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--segments", default="segments_labeled.npz")
    parser.add_argument("--embeddings", default="reve_embeddings.npz")
    parser.add_argument("--signal-key")
    parser.add_argument("--embedding-key")
    parser.add_argument("--fs", type=float, default=200)
    parser.add_argument("--acf-max-lag", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--out", default="results_c")

    args = parser.parse_args()

    if (
        args.batch_size < 1
        or not np.isfinite(args.fs)
        or args.fs <= 0
        or any(not np.isfinite(v) or v <= 0 for v in args.alphas)
        or (args.acf_max_lag is not None and args.acf_max_lag < 1)
    ):
        parser.error("Paramètres numériques invalides.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if any(out.iterdir()):
        raise FileExistsError(
            f"{out} non vide : choisir un nouveau --out."
        )

    print("Lecture des en-têtes NPZ...", flush=True)

    signal_key = choose_key(args.segments, args.signal_key, "eeg")
    embedding_key = choose_key(
        args.embeddings, args.embedding_key, "embedding"
    )

    shape, _, _ = info(args.segments, signal_key)
    embedding_shape, _, _ = info(args.embeddings, embedding_key)

    if len(shape) == 2:
        n, time = shape
        channels = 1
    elif len(shape) == 3:
        n, channels, time = shape
    else:
        raise ValueError("EEG attendu : N×C×T ou N×T.")

    if (
        min(n, channels) < 1
        or time < 64
        or len(embedding_shape) < 2
        or embedding_shape[0] != n
        or min(embedding_shape[1:]) < 1
    ):
        raise ValueError(
            "Formes incompatibles. Au moins 64 points sont requis "
            "pour disposer de deux échelles DFA."
        )

    requested_lag = (
        args.acf_max_lag
        if args.acf_max_lag is not None
        else max(1, int(2 * args.fs))
    )
    max_lag = min(time // 2, requested_lag)

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
        raise ValueError("Labels EEG et embeddings non alignés.")

    features = np.empty((n, channels, 7))
    qc = []

    print(
        f"Extraction C : EEG={shape}, "
        f"tau_max={max_lag} échantillons",
        flush=True,
    )

    for start, batch in batches(
        args.segments, signal_key, args.batch_size
    ):
        batch = np.asarray(batch, dtype=np.float64)

        if batch.ndim == 2:
            batch = batch[:, None, :]

        for i in range(len(batch)):
            for ch in range(channels):
                values, raw_c4 = extract_channel(
                    batch[i, ch], max_lag
                )
                features[start + i, ch] = values

                row = {
                    "row_index": start + i,
                    "channel": ch,
                    "signal_std": float(batch[i, ch].std()),
                    "C004_before_clip": raw_c4,
                }

                row.update({
                    name[:4] + "_undefined": not np.isfinite(values[j])
                    for j, name in enumerate(NAMES)
                })

                qc.append(row)

        print(
            f"EEG : {start + len(batch)}/{n}",
            end="\r", flush=True,
        )

    print(flush=True)
    pd.DataFrame(qc).to_csv(out / "c_qc.csv", index=False)

    np.savez_compressed(
        out / "c_features.npz",
        features=features,
        feature_names=np.asarray(NAMES),
        labels=labels if labels is not None else np.array([]),
        row_index=np.arange(n),
        sampling_frequency=args.fs,
        acf_max_lag=max_lag,
    )

    with (out / "c_features.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["segment", "channel", "label"] + NAMES)

        for i in range(n):
            for ch in range(channels):
                writer.writerow([
                    i, ch, labels[i] if labels is not None else "",
                    *features[i, ch],
                ])

    embeddings = np.empty((n, embedding_shape[-1]))

    print("Lecture et pooling des embeddings par lots...", flush=True)

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

    valid = (
        np.isfinite(embeddings).all(axis=1)
        & np.isfinite(features).all(axis=(1, 2))
    )
    original_rows = np.flatnonzero(valid)

    if len(original_rows) < 14:
        raise ValueError(
            "Moins de 14 essais finis : examiner c_qc.csv."
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
    f = features[valid]

    x = StandardScaler().fit(e[train]).transform(e)
    trainval = np.concatenate([train, validation])

    results = []
    channel_results = []
    predictions = np.empty((len(test), channels, 7))

    print(
        f"Train={len(train)}, validation={len(validation)}, "
        f"test={len(test)}, exclus={int((~valid).sum())}",
        flush=True,
    )

    for j, name in enumerate(NAMES):
        y = f[:, :, j]
        y_scaler = StandardScaler().fit(y[train])
        scaled_y = y_scaler.transform(y)
        scores = []

        for alpha in args.alphas:
            model = Ridge(alpha=alpha).fit(
                x[train], scaled_y[train]
            )
            prediction = predict(model, x[validation])

            scores.append(average(
                raw_r2(scaled_y[validation], prediction)
            ))

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
        predictions[:, :, j] = prediction

        r2 = raw_r2(y[test], prediction)

        results.append({
            "Feature": name,
            "Best_Alpha": alpha,
            "Validation_R2": scores[best],
            "Test_R2": average(r2),
            "Defined_Test_Channels": int(np.isfinite(r2).sum()),
            "Status": status,
        })

        for ch in range(channels):
            train_std = float(y[train, ch].std())
            test_std = float(y[test, ch].std())

            channel_results.append({
                "Feature": name,
                "Channel": ch,
                "Test_R2": r2[ch],
                "Test_RMSE": float(np.sqrt(np.mean(
                    (y[test, ch] - prediction[:, ch]) ** 2
                ))),
                "Test_MAE": float(np.mean(
                    np.abs(y[test, ch] - prediction[:, ch])
                )),
                "Train_Std": train_std,
                "TrainVal_Std": float(y[trainval, ch].std()),
                "Test_Std": test_std,
                "Low_Variance_Train": bool(
                    train_std <= 1e-8 + 1e-6 * abs(y[train, ch].mean())
                ),
                "Low_Variance_Test": bool(
                    test_std <= 1e-8 + 1e-6 * abs(y[test, ch].mean())
                ),
                "Undefined_Test_R2": not np.isfinite(r2[ch]),
            })

        print(
            f"{name:28s} alpha={alpha:g} "
            f"val R²={scores[best]:.4f} "
            f"test R²={average(r2):.4f}",
            flush=True,
        )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        out / "reve_c_ridge_results.csv", index=False
    )

    pd.DataFrame(channel_results).to_csv(
        out / "reve_c_channel_results.csv", index=False
    )

    np.savez_compressed(
        out / "test_predictions.npz",
        row_index=original_rows[test],
        y_true=f[test],
        y_pred=predictions,
        feature_names=np.asarray(NAMES),
    )

    config = vars(args) | {
        "acf_max_lag_effective": max_lag,
        "n_total": n,
        "n_valid": len(original_rows),
        "n_channels": channels,
        "source": "https://arxiv.org/pdf/2605.11410, eq.27-33",
        "conventions": (
            "stable ordinal ties; raw finite differences; "
            "positive log terms only; DFA forward complete windows; "
            "centered zero-padded FFT ACF, biased normalization"
        ),
        "acf_default": (
            "min(T//2, int(2*fs)); implementation choice; "
            "numeric tau_max unspecified in paper"
        ),
        "split": "random trial-level 70/15/15; subjects may overlap",
        "scaling": "fit train only; Ridge refit train+validation",
        "alignment": "original row order required; available labels checked",
        "r2": "raw; constant targets undefined; no clipping",
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
