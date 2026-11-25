import argparse
import csv
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
    "X001_MI_DeltaBeta",
    "X002_MI_ThetaGamma",
    "X003_MI_AlphaGamma",
    "X004_AAC_ThetaGamma",
    "X005_AAC_DeltaGamma",
]

BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]


def modulation_index(phase, amplitude, empty_policy="nan"):
    # Ramener +pi à -pi pour conserver tous les échantillons.
    phase = (phase + np.pi) % (2 * np.pi) - np.pi

    index = np.floor(
        (phase + np.pi) * 18 / (2 * np.pi)
    ).astype(int)
    index = np.clip(index, 0, 17)

    counts = np.bincount(index, minlength=18)
    totals = np.bincount(
        index, weights=amplitude, minlength=18
    )

    empty = int((counts == 0).sum())

    # Une moyenne conditionnelle n'est pas définie dans un bin vide.
    if empty and empty_policy == "nan":
        return np.nan, empty

    means = np.divide(
        totals, counts,
        out=np.zeros(18),
        where=counts > 0,
    )

    total = means.sum()

    if not np.isfinite(total) or total <= 0:
        return np.nan, empty

    probability = means / total
    positive = probability > 0

    value = np.sum(
        probability[positive]
        * np.log(18 * probability[positive])
    ) / np.log(18)

    return float(np.clip(value, 0, 1)), empty


def envelope_correlation(a, b, amplitude_eps, relative_eps):
    # Éviter une corrélation artificielle entre enveloppes constantes.
    if (
        a.std() <= amplitude_eps + relative_eps * abs(a.mean())
        or b.std() <= amplitude_eps + relative_eps * abs(b.mean())
    ):
        return np.nan

    aa = a - a.mean()
    bb = b - b.mean()

    value = np.dot(aa, bb) / np.sqrt(
        np.dot(aa, aa) * np.dot(bb, bb)
    )

    return float(np.clip(value, -1, 1))


def extract_channel(
    x, fs, lowpass, amplitude_eps, relative_eps, empty_policy
):
    if not np.isfinite(x).all():
        return np.full(5, np.nan), {"nonfinite_signal": True}

    frequencies = np.fft.rfftfreq(len(x), 1 / fs)
    spectrum = np.fft.rfft(x)
    cap = min(45.0, lowpass, fs / 2)

    phases = []
    amplitudes = []

    for name, low, high in BANDS:
        mask = (
            (frequencies >= low)
            & (frequencies < min(high, cap))
        )

        if not mask.any():
            raise ValueError(
                f"Bande {name} sans bin FFT : "
                "vérifier fs, lowpass et durée."
            )

        filtered = np.fft.irfft(
            spectrum * mask, n=len(x)
        )
        analytic = hilbert(filtered)

        phases.append(np.angle(analytic))
        amplitudes.append(np.abs(analytic))

    usable = [
        amplitude.mean() > amplitude_eps
        for amplitude in amplitudes
    ]

    values = []
    qc = {"nonfinite_signal": False}

    # X001 : delta -> beta
    # X002 : theta -> gamma
    # X003 : alpha -> gamma
    for j, (low, high) in enumerate(
        [(0, 3), (1, 4), (2, 4)], 1
    ):
        if usable[low] and usable[high]:
            value, empty = modulation_index(
                phases[low],
                amplitudes[high],
                empty_policy,
            )
        else:
            value, empty = np.nan, -1

        values.append(value)
        qc[f"X00{j}_empty_phase_bins"] = empty

    # X004 : theta-gamma ; X005 : delta-gamma.
    for low, high in [(1, 4), (0, 4)]:
        if usable[low] and usable[high]:
            value = envelope_correlation(
                amplitudes[low],
                amplitudes[high],
                amplitude_eps,
                relative_eps,
            )
        else:
            value = np.nan

        values.append(value)

    for i, (name, _, _) in enumerate(BANDS):
        qc[name + "_envelope_mean"] = float(
            amplitudes[i].mean()
        )
        qc[name + "_envelope_std"] = float(
            amplitudes[i].std()
        )

    return np.array(values), qc


def header(handle):
    version = np.lib.format.read_magic(handle)

    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }

    if version not in readers:
        raise ValueError(
            f"Version NPY non prise en charge : {version}"
        )

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
    """Lecture séquentielle du NPZ sans chargement complet."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + ".npy") as handle:
            shape, fortran, dtype = header(handle)

            if fortran or dtype.kind not in "iuf" or not shape:
                raise ValueError(
                    "Tableau numérique C-order requis."
                )

            row_bytes = int(np.prod(shape[1:])) * dtype.itemsize

            for start in range(0, shape[0], batch_size):
                count = min(batch_size, shape[0] - start)
                raw = handle.read(count * row_bytes)

                if len(raw) != count * row_bytes:
                    raise ValueError("Tableau NPZ tronqué.")

                yield start, np.frombuffer(
                    raw, dtype=dtype
                ).reshape((count,) + shape[1:])


def labels_from(path):
    # N'utiliser que vos propres NPZ de confiance.
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
        "--envelope-rtol", type=float, default=1e-8
    )
    parser.add_argument(
        "--empty-bins", choices=["nan", "zero"], default="nan"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--out", default=None)

    args = parser.parse_args()
    paths = Paths(vars(args).pop('config'))
    args.segments = args.segments or str(paths.input("segments"))
    args.embeddings = args.embeddings or str(paths.input("embeddings"))
    args.out = args.out or str(paths.family_result("X"))

    if (
        args.batch_size < 1
        or not np.isfinite(args.fs)
        or args.fs <= 0
        or any(
            not np.isfinite(v) or v <= 0
            for v in args.alphas
        )
        or not np.isfinite(args.lowpass)
        or args.lowpass <= 30
        or not np.isfinite(args.amplitude_eps)
        or args.amplitude_eps < 0
        or not np.isfinite(args.envelope_rtol)
        or args.envelope_rtol < 0
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

    if len(shape) == 2:
        n, time = shape
        channels = 1
    elif len(shape) == 3:
        n, channels, time = shape
    else:
        raise ValueError("EEG attendu : N×C×T ou N×T.")

    if (
        min(n, channels) < 1
        or time < 3
        or len(embedding_shape) < 2
        or embedding_shape[0] != n
        or min(embedding_shape[1:]) < 1
    ):
        raise ValueError("Formes EEG/embeddings incompatibles.")

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

    features = np.empty((n, channels, len(NAMES)))
    qc = []

    print(
        f"Extraction X : EEG={shape}, "
        f"fs={args.fs} Hz, 18 bins de phase",
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
                values, diagnostic = extract_channel(
                    batch[i, ch],
                    args.fs,
                    args.lowpass,
                    args.amplitude_eps,
                    args.envelope_rtol,
                    args.empty_bins,
                )

                features[start + i, ch] = values

                row = {
                    "row_index": start + i,
                    "channel": ch,
                    "signal_std": float(batch[i, ch].std()),
                    **diagnostic,
                }

                row.update({
                    name[:4] + "_undefined":
                        not np.isfinite(values[j])
                    for j, name in enumerate(NAMES)
                })

                qc.append(row)

        print(
            f"EEG : {start + len(batch)}/{n}",
            end="\r", flush=True,
        )

    print(flush=True)
    pd.DataFrame(qc).to_csv(out / "x_qc.csv", index=False)

    np.savez_compressed(
        out / "x_features.npz",
        features=features,
        feature_names=np.asarray(NAMES),
        labels=labels if labels is not None else np.array([]),
        row_index=np.arange(n),
        sampling_frequency=args.fs,
        phase_bins=18,
        empty_bin_policy=args.empty_bins,
        band_edges_hz=np.asarray([
            (low, min(high, args.lowpass, args.fs / 2))
            for _, low, high in BANDS
        ]),
    )

    with (out / "x_features.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["segment", "channel", "label"] + NAMES)

        for i in range(n):
            for ch in range(channels):
                writer.writerow([
                    i, ch,
                    labels[i] if labels is not None else "",
                    *features[i, ch],
                ])

    embeddings = np.empty((n, embedding_shape[-1]))

    print(
        "Lecture et pooling des embeddings par lots...",
        flush=True,
    )

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
            "Moins de 14 essais finis : examiner x_qc.csv."
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
    predictions = np.empty(
        (len(test), channels, len(NAMES))
    )

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
                    train_std <=
                    1e-8 + 1e-6 * abs(y[train, ch].mean())
                ),
                "Low_Variance_Test": bool(
                    test_std <=
                    1e-8 + 1e-6 * abs(y[test, ch].mean())
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
        out / "reve_x_ridge_results.csv", index=False
    )
    pd.DataFrame(channel_results).to_csv(
        out / "reve_x_channel_results.csv", index=False
    )

    np.savez_compressed(
        out / "test_predictions.npz",
        row_index=original_rows[test],
        y_true=f[test],
        y_pred=predictions,
        feature_names=np.asarray(NAMES),
    )

    config = vars(args) | {
        "n_total": n,
        "n_valid": len(original_rows),
        "n_channels": channels,
        "source": "https://arxiv.org/pdf/2605.11410, eq.34-37",
        "conventions": (
            "rectangular FFT bandpass + Hilbert; "
            "18 phase bins; mean amplitude per bin; "
            "normalized Tort MI; envelope Pearson correlation"
        ),
        "undefined_policy": (
            "weak band amplitude, empty phase bins with nan policy, "
            "or near-constant AAC envelope -> NaN; "
            "rows excluded from regression"
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
