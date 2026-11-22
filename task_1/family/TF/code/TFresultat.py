import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "TF001_SubbandEntropy",
    "TF002_DetailVarianceL1",
    "TF003_DetailVarianceL2",
    "TF004_DetailVarianceL3",
    "TF005_DetailVarianceL4",
    "TF006_DetailVarianceL5",
    "TF007_EnvelopeCVDelta",
    "TF008_EnvelopeCVTheta",
    "TF009_EnvelopeCVAlpha",
    "TF010_EnvelopeCVBeta",
    "TF011_EnvelopeCVGamma",
]

BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]


def haar_details(x):
    """Cinq niveaux Haar, sans padding, normalisation 1/sqrt(2)."""
    current = x
    details = []

    for _ in range(5):
        if current.shape[-1] < 2:
            raise ValueError("Au moins 32 points nécessaires.")

        # Écarter le dernier coefficient si la longueur est impaire.
        n = current.shape[-1] - current.shape[-1] % 2
        even = current[..., :n:2]
        odd = current[..., 1:n:2]

        details.append((even - odd) / np.sqrt(2))
        current = (even + odd) / np.sqrt(2)

    return details, current


def extract_features(x, fs, lowpass, envelope_eps):
    details, approximation = haar_details(x)

    # Énergies MOYENNES par coefficient, pas sommes des énergies.
    energies = np.stack(
        [np.mean(d ** 2, axis=-1) for d in details]
        + [np.mean(approximation ** 2, axis=-1)],
        axis=-1,
    )

    total = energies.sum(axis=-1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        probability = energies / total
        terms = np.where(
            probability > 0,
            probability * np.log(probability),
            0,
        )
        entropy = -terms.sum(axis=-1) / np.log(6)

    entropy = np.where(total[..., 0] > 0, entropy, np.nan)

    variances = np.stack(
        [np.var(d, axis=-1, ddof=0) for d in details],
        axis=-1,
    )

    frequencies = np.fft.rfftfreq(x.shape[-1], d=1 / fs)
    spectrum = np.fft.rfft(x, axis=-1)
    cap = min(45.0, lowpass, fs / 2)

    envelope_means = []
    envelope_cvs = []

    for _, low, high in BANDS:
        mask = (
            (frequencies >= low)
            & (frequencies < min(high, cap))
        )

        if not mask.any():
            raise ValueError(
                f"Bande [{low}, {min(high, cap)}) sans bin FFT."
            )

        filtered = np.fft.irfft(
            spectrum * mask,
            n=x.shape[-1],
            axis=-1,
        )

        envelope = np.abs(hilbert(filtered, axis=-1))
        mean = envelope.mean(axis=-1)
        std = envelope.std(axis=-1, ddof=0)

        cv = np.full_like(mean, np.nan)
        np.divide(
            std,
            mean,
            out=cv,
            where=mean > envelope_eps,
        )

        envelope_means.append(mean)
        envelope_cvs.append(cv)

    features = np.concatenate(
        [
            entropy[..., None],
            variances,
            np.stack(envelope_cvs, axis=-1),
        ],
        axis=-1,
    )

    invalid = ~np.isfinite(x).all(axis=-1)
    features[invalid] = np.nan

    return features, energies, np.stack(envelope_means, axis=-1)


def read_header(handle):
    version = np.lib.format.read_magic(handle)

    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }

    if version not in readers:
        raise ValueError(f"Version NPY non prise en charge : {version}")

    return readers[version](handle)


def array_header(path, key):
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + ".npy") as handle:
            return read_header(handle)


def stream_rows(path, key, batch_size):
    """Lire le tableau NPZ par lots sans le charger entièrement."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + ".npy") as handle:
            shape, fortran, dtype = read_header(handle)

            if fortran or dtype.kind not in "iuf" or not shape:
                raise ValueError(
                    "Le tableau doit être numérique et stocké en ordre C."
                )

            row_size = int(np.prod(shape[1:])) * dtype.itemsize

            for start in range(0, shape[0], batch_size):
                count = min(batch_size, shape[0] - start)
                size = count * row_size
                raw = handle.read(size)

                if len(raw) != size:
                    raise ValueError("Tableau NPZ tronqué.")

                batch = np.frombuffer(raw, dtype=dtype).reshape(
                    (count,) + shape[1:]
                )

                yield start, batch


def select_key(path, explicit, kind):
    with zipfile.ZipFile(path) as archive:
        keys = [
            name[:-4]
            for name in archive.namelist()
            if name.endswith(".npy")
        ]

    if explicit:
        if explicit not in keys:
            raise ValueError(f"Clé absente : {explicit}. Clés : {keys}")
        return explicit

    if kind == "embedding" and "embeddings" in keys:
        return "embeddings"

    candidates = []

    for key in keys:
        shape, _, dtype = array_header(path, key)

        if dtype.kind not in "iuf":
            continue

        if kind == "eeg":
            if (
                key.lower() in ("segments", "x", "eeg", "signals", "data")
                and len(shape) in (2, 3)
            ):
                candidates.append(key)

        elif len(shape) >= 2:
            candidates.append(key)

    if len(candidates) != 1:
        raise ValueError(
            f"Clé {kind} ambiguë : {candidates}. "
            "Préciser --signal-key ou --embedding-key."
        )

    return candidates[0]


def load_labels(path):
    # Utiliser uniquement vos propres fichiers NPZ de confiance.
    with np.load(path, allow_pickle=True) as data:
        keys = [
            key for key in data.files
            if key.lower() in ("y", "labels", "label", "targets")
        ]

        if len(keys) > 1:
            raise ValueError(f"Plusieurs clés de labels : {keys}")

        if not keys:
            return None

        labels = np.asarray(data[keys[0]])

    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]

    if labels.ndim != 1:
        raise ValueError("Les labels doivent être un vecteur.")

    return labels


def r2_channels(y, prediction):
    """R² négatifs conservés ; cibles constantes = NaN."""
    result = np.full(y.shape[1], np.nan)

    if len(y) >= 2:
        denominator = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        valid = (np.ptp(y, axis=0) > 0) & (denominator > 0)

        result[valid] = 1 - (
            ((y[:, valid] - prediction[:, valid]) ** 2).sum(axis=0)
            / denominator[valid]
        )

    return result


def mean_defined(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else np.nan


def predict_2d(model, x):
    return np.asarray(model.predict(x)).reshape(len(x), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", default="segments_labeled.npz")
    parser.add_argument("--embeddings", default="reve_embeddings.npz")
    parser.add_argument("--signal-key", default=None)
    parser.add_argument("--embedding-key", default=None)
    parser.add_argument("--fs", type=float, default=200)
    parser.add_argument("--lowpass", type=float, default=45)
    parser.add_argument("--envelope-eps", type=float, default=1e-12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--out", default="results_tf")
    args = parser.parse_args()

    if (
        args.batch_size < 1
        or not np.isfinite(args.fs)
        or args.fs <= 0
        or not np.isfinite(args.lowpass)
        or args.lowpass <= 30
        or not np.isfinite(args.envelope_eps)
        or args.envelope_eps < 0
        or any(not np.isfinite(a) or a <= 0 for a in args.alphas)
    ):
        parser.error("Paramètres numériques invalides.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if any(out.iterdir()):
        raise FileExistsError(
            f"{out} contient déjà des fichiers. Choisir un nouveau --out."
        )

    print("Lecture des en-têtes NPZ...", flush=True)

    signal_key = select_key(args.segments, args.signal_key, "eeg")
    embedding_key = select_key(
        args.embeddings, args.embedding_key, "embedding"
    )

    shape, _, _ = array_header(args.segments, signal_key)
    embedding_shape, _, _ = array_header(
        args.embeddings, embedding_key
    )

    if len(shape) == 2:
        n, time = shape
        channels = 1
    elif len(shape) == 3:
        n, channels, time = shape
    else:
        raise ValueError(f"Forme EEG incompatible : {shape}")

    if (
        min(n, channels, time) < 1
        or time < 32
        or len(embedding_shape) < 2
        or embedding_shape[0] != n
        or min(embedding_shape[1:]) < 1
    ):
        raise ValueError(
            f"Formes incompatibles : EEG={shape}, "
            f"embeddings={embedding_shape}. "
            "Au moins 32 points EEG sont requis."
        )

    labels = load_labels(args.segments)
    embedding_labels = load_labels(args.embeddings)

    for values in (labels, embedding_labels):
        if values is not None and len(values) != n:
            raise ValueError("Nombre de labels incompatible.")

    if (
        labels is not None
        and embedding_labels is not None
        and not np.array_equal(labels, embedding_labels)
    ):
        raise ValueError("Labels EEG et embeddings non alignés.")

    print(
        f"EEG : {shape} ; embeddings : {embedding_shape} ; "
        f"fs={args.fs} Hz",
        flush=True,
    )

    features = np.empty((n, channels, 11), dtype=np.float64)
    qc = []

    print("Extraction TF001–TF011 par lots...", flush=True)

    for start, batch in stream_rows(
        args.segments, signal_key, args.batch_size
    ):
        batch = np.asarray(batch, dtype=np.float64)

        if batch.ndim == 2:
            batch = batch[:, None, :]

        values, energies, envelope_means = extract_features(
            batch, args.fs, args.lowpass, args.envelope_eps
        )

        features[start:start + len(batch)] = values

        for i in range(len(batch)):
            qc.append({
                "row_index": start + i,
                "nonfinite_signal_channels": int(
                    (~np.isfinite(batch[i]).all(axis=-1)).sum()
                ),
                "zero_haar_energy_channels": int(
                    (energies[i].sum(axis=-1) <= 0).sum()
                ),
                "undefined_envelope_cv": int(
                    (
                        ~np.isfinite(envelope_means[i])
                        | (envelope_means[i] <= args.envelope_eps)
                    ).sum()
                ),
            })

        print(f"EEG : {start + len(batch)}/{n}", end="\r", flush=True)

    print(flush=True)
    pd.DataFrame(qc).to_csv(out / "tf_qc.csv", index=False)

    np.savez_compressed(
        out / "tf_features.npz",
        features=features,
        labels=labels if labels is not None else np.array([]),
        feature_names=np.asarray(FEATURE_NAMES),
        sampling_frequency=args.fs,
        row_index=np.arange(n),
        band_names=np.asarray([b[0] for b in BANDS]),
        band_edges_hz=np.asarray([
            (low, min(high, args.lowpass, args.fs / 2))
            for _, low, high in BANDS
        ]),
        envelope_mean_threshold=args.envelope_eps,
        haar_levels=5,
        haar_odd_policy="drop final unpaired coefficient at each odd level",
    )

    with (out / "tf_features.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["segment", "channel", "label"] + FEATURE_NAMES)

        for i in range(n):
            for ch in range(channels):
                writer.writerow([
                    i, ch, labels[i] if labels is not None else "",
                    *features[i, ch],
                ])

    embeddings = np.empty(
        (n, embedding_shape[-1]), dtype=np.float64
    )

    print("Lecture et pooling des embeddings par lots...", flush=True)

    for start, batch in stream_rows(
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
            "Moins de 14 essais finis. Examiner tf_qc.csv."
        )

    e = embeddings[valid]
    f = features[valid]

    train, remaining = train_test_split(
        np.arange(len(e)),
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

    print(
        f"Train={len(train)}, validation={len(validation)}, "
        f"test={len(test)}, exclus={int((~valid).sum())}",
        flush=True,
    )

    x_scaler = StandardScaler().fit(e[train])
    x = x_scaler.transform(e)
    trainval = np.concatenate([train, validation])

    results = []
    channel_results = []
    predictions = np.full((len(test), channels, 11), np.nan)

    print("Régressions Ridge...", flush=True)

    for j, name in enumerate(FEATURE_NAMES):
        y = f[:, :, j]
        y_scaler = StandardScaler().fit(y[train])
        scaled_y = y_scaler.transform(y)

        validation_scores = []

        for alpha in args.alphas:
            model = Ridge(alpha=alpha).fit(
                x[train], scaled_y[train]
            )
            prediction = predict_2d(model, x[validation])
            score = mean_defined(
                r2_channels(scaled_y[validation], prediction)
            )
            validation_scores.append(score)

        if np.isfinite(validation_scores).any():
            best = int(np.argmax(np.where(
                np.isfinite(validation_scores),
                validation_scores,
                -np.inf,
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
            predict_2d(model, x[test])
        )
        predictions[:, :, j] = prediction

        raw_r2 = r2_channels(y[test], prediction)
        mean_r2 = mean_defined(raw_r2)

        results.append({
            "Feature": name,
            "Best_Alpha": alpha,
            "Validation_R2": validation_scores[best],
            "Test_R2": mean_r2,
            "Defined_Test_Channels": int(np.isfinite(raw_r2).sum()),
            "Status": status,
        })

        for ch in range(channels):
            train_std = float(y[train, ch].std())
            test_std = float(y[test, ch].std())

            channel_results.append({
                "Feature": name,
                "Channel": ch,
                "Test_R2": raw_r2[ch],
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
                "Undefined_Test_R2": not np.isfinite(raw_r2[ch]),
            })

        print(
            f"{name:26s} alpha={alpha:g} "
            f"val R²={validation_scores[best]:.4f} "
            f"test R²={mean_r2:.4f}",
            flush=True,
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        out / "reve_tf_ridge_results.csv", index=False
    )

    pd.DataFrame(channel_results).to_csv(
        out / "reve_tf_channel_results.csv", index=False
    )

    np.savez_compressed(
        out / "test_predictions.npz",
        row_index=original_rows[test],
        y_true=f[test],
        y_pred=predictions,
        feature_names=np.asarray(FEATURE_NAMES),
    )

    config = vars(args) | {
        "n_total": n,
        "n_valid": len(original_rows),
        "n_channels": channels,
        "signal_key": signal_key,
        "embedding_key": embedding_key,
        "split": "random trial-level 70/15/15; subjects may overlap",
        "alignment": "original row order required; available labels checked",
        "r2": "raw; constant targets undefined; no clipping",
        "standardization": "fit train; final Ridge refit train+validation",
        "haar": "5 levels; mean coefficient energies; drop odd last item",
        "envelope": "rectangular FFT mask, irfft, Hilbert amplitude",
        "undefined_cv": "mean envelope <= envelope_eps produces NaN",
        "source": "https://arxiv.org/pdf/2605.11410, equations 24-26",
        "completed": True,
    }

    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== RÉSULTATS ==========")
    print(result_df.to_string(index=False))
    print(f"\nFichiers enregistrés dans : {out.resolve()}")


if __name__ == "__main__":
    main()
