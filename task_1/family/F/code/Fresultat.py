"""Extraction des 16 features spectrales F + régression Ridge depuis REVE.
Exemple, depuis le dossier data :
python Fresultat.py --out results_spectral
Attention : découpage aléatoire par essai, pas une évaluation indépendante par sujet.
"""
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

FEATURE_NAMES = [
    'F001_LogDelta', 'F002_LogTheta', 'F003_LogAlpha', 'F004_LogBeta', 'F005_LogGamma',
    'F006_RelativeDelta', 'F007_RelativeTheta', 'F008_RelativeAlpha', 'F009_RelativeBeta', 'F010_RelativeGamma',
    'F011_LogThetaBeta', 'F012_LogDeltaAlpha', 'F013_LogThetaAlpha',
    'F014_SpectralEntropy', 'F015_SpectralCentroid', 'F016_SpectralEdge95',
]
BANDS = [('delta', .5, 4.), ('theta', 4., 8.), ('alpha', 8., 13.), ('beta', 13., 30.), ('gamma', 30., 45.)]


def extract_features(x, fs=200, lowpass=45., eps=1e-12):
    """Annexe A, équations 16–23 : |rfft(x)|², sans normalisation ni fenêtre.
    Retour : valeurs (...,16), énergies par bande (...,5).
    Les canaux d'énergie canonique nulle sont marqués NaN, pas inventés à zéro.
    """
    freq = np.fft.rfftfreq(x.shape[-1], d=1 / fs)
    spectrum = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    cap = min(45., lowpass, fs / 2)
    masks = [(freq >= low) & (freq < min(high, cap)) for _, low, high in BANDS]
    if any(not mask.any() for mask in masks):
        raise ValueError('Au moins une bande ne contient aucun bin FFT. Vérifier fs, lowpass et durée des essais.')
    union = np.logical_or.reduce(masks)
    if union.sum() < 2:
        raise ValueError('Au moins deux bins canoniques nécessaires pour F014')
    powers = np.stack([spectrum[..., mask].sum(axis=-1) for mask in masks], axis=-1)
    s = spectrum[..., union]
    total = powers.sum(axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative = powers / total
        probability = s / total
        logs = np.log(np.maximum(powers, eps))
        entropy = -np.sum(np.where(probability > 0, probability * np.log(probability), 0), axis=-1) / np.log(union.sum())
        centroid = np.sum(freq[union] * probability, axis=-1)
    cumulative = np.cumsum(probability, axis=-1)
    # Premier bin dont la masse cumulée atteint 95 %.
    index = np.argmax(cumulative >= .95, axis=-1)
    edge = freq[union][index]
    ratios = np.stack([logs[..., 1] - logs[..., 3], logs[..., 0] - logs[..., 2], logs[..., 1] - logs[..., 2]], axis=-1)
    features = np.concatenate([logs, relative, ratios, np.stack([entropy, centroid, edge], axis=-1)], axis=-1)
    invalid = (total[..., 0] <= 0) | ~np.isfinite(total[..., 0]) | ~np.isfinite(x).all(axis=-1)
    features[invalid] = np.nan
    return features, powers


def array_header(path, key):
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + '.npy') as handle:
            version = np.lib.format.read_magic(handle)
            reader = { (1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0 }.get(version)
            if reader is None:
                raise ValueError(f'Format NPY {version} non pris en charge par le lecteur par lots')
            shape, fortran, dtype = reader(handle)
    return shape, fortran, dtype


def stream_rows(path, key, batch_size):
    # Lecture séquentielle à travers le ZIP, sans charger/décompresser tout le tableau.
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + '.npy') as handle:
            version = np.lib.format.read_magic(handle)
            reader = { (1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0 }.get(version)
            if reader is None:
                raise ValueError(f'Format NPY {version} non pris en charge')
            shape, fortran, dtype = reader(handle)
            if fortran or dtype.kind not in 'iuf' or not shape:
                raise ValueError('Le lecteur par lots exige un tableau numérique C-order. Réexporter ce tableau si nécessaire.')
            row_values = int(np.prod(shape[1:]))
            for start in range(0, shape[0], batch_size):
                count = min(batch_size, shape[0] - start)
                size = count * row_values * dtype.itemsize
                raw = handle.read(size)
                if len(raw) != size:
                    raise ValueError('Tableau NPZ tronqué')
                yield start, np.frombuffer(raw, dtype=dtype).reshape((count,) + shape[1:])


def select_stream_key(path, explicit, kind):
    with zipfile.ZipFile(path) as archive:
        keys = [name[:-4] for name in archive.namelist() if name.endswith('.npy')]
    if explicit:
        if explicit not in keys:
            raise ValueError(f'Clé {explicit!r} absente : {keys}')
        return explicit
    if kind == 'embedding' and 'embeddings' in keys:
        return 'embeddings'
    candidates = []
    for key in keys:
        shape, _, dtype = array_header(path, key)
        if dtype.kind not in 'iuf':
            continue
        if kind == 'eeg' and key.lower() in ('segments', 'x', 'eeg', 'signals', 'data') and len(shape) in (2, 3):
            candidates.append(key)
        elif kind == 'embedding' and len(shape) >= 2:
            candidates.append(key)
    if len(candidates) != 1:
        raise ValueError(f'Clé {kind} ambiguë : {candidates}. Préciser --signal-key ou --embedding-key.')
    return candidates[0]


def r2_per_channel(y, prediction):
    # Conserver les R² négatifs. Les cibles constantes ont un R² indéfini.
    r2 = np.full(y.shape[1], np.nan)
    if len(y) >= 2:
        denom = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        good = (np.ptp(y, axis=0) > 0) & (denom > 0)
        r2[good] = 1 - ((y[:, good] - prediction[:, good]) ** 2).sum(axis=0) / denom[good]
    return r2


def defined_mean(v):
    return float(np.mean(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan


def prediction_2d(model, x):
    return np.asarray(model.predict(x)).reshape(len(x), -1)


def load_labels(data):
    candidates = [k for k in data.files if k.lower() in ('y', 'labels', 'label', 'targets')]
    if len(candidates) > 1:
        raise ValueError(f'Plusieurs clés de labels : {candidates}')
    if not candidates:
        return None
    labels = np.asarray(data[candidates[0]])
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise ValueError('Les labels doivent être un vecteur de longueur N')
    return labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segments', default='segments_labeled.npz')
    p.add_argument('--embeddings', default='reve_embeddings.npz')
    p.add_argument('--signal-key', default=None)
    p.add_argument('--embedding-key', default=None)
    p.add_argument('--fs', type=float, default=200)
    p.add_argument('--lowpass', type=float, default=45., help='Plafond des bandes en Hz ; ne filtre pas le signal')
    p.add_argument('--eps', type=float, default=1e-12, help='Plancher des énergies uniquement pour logarithmes')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--alphas', nargs='+', type=float, default=[.1, 1., 10., 100.])
    p.add_argument('--out', default='results_spectral')
    a = p.parse_args()
    if a.batch_size < 1 or not np.isfinite(a.eps) or a.eps <= 0 or not np.isfinite(a.lowpass) or a.lowpass <= 30 or not np.isfinite(a.fs) or a.fs <= 0 or any(not np.isfinite(v) or v <= 0 for v in a.alphas):
        p.error('fs, eps et alphas finis > 0 ; lowpass > 30 ; batch-size >= 1 requis')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'{out} contient déjà des fichiers. Choisissez un autre --out.')
    print('Lecture des en-têtes NPZ (sans charger les grands tableaux)...', flush=True)
    signal_key = select_stream_key(a.segments, a.signal_key, 'eeg')
    embedding_key = select_stream_key(a.embeddings, a.embedding_key, 'embedding')
    shape, _, _ = array_header(a.segments, signal_key)
    eshape, _, _ = array_header(a.embeddings, embedding_key)
    if len(shape) == 2:
        n, time = shape
        channels = 1
    elif len(shape) == 3:
        n, channels, time = shape
    else:
        raise ValueError(f'EEG incompatible : {shape}')
    if min(n, channels, time) < 1 or time < 3 or len(eshape) < 2 or eshape[0] != n or eshape[-1] < 1:
        raise ValueError(f'Formes incompatibles : EEG={shape}, embeddings={eshape}')
    # Seuls les petits vecteurs de labels sont chargés directement.
    with np.load(a.segments, allow_pickle=True) as data:
        labels = load_labels(data)
    with np.load(a.embeddings, allow_pickle=True) as data:
        embedding_labels = load_labels(data)
    if labels is not None and len(labels) != n or embedding_labels is not None and len(embedding_labels) != n:
        raise ValueError('Nombre de labels différent du nombre de segments')
    if labels is not None and embedding_labels is not None and not np.array_equal(labels, embedding_labels):
        raise ValueError('Labels EEG/embeddings non alignés')
    print(f'EEG={shape}, embeddings={eshape}, fs={a.fs} Hz ; lots de {a.batch_size} essais', flush=True)
    f = np.empty((n, channels, len(FEATURE_NAMES)), dtype=np.float64)
    qc = []
    print('Extraction FFT F001–F016 par lots...', flush=True)
    for start, batch in stream_rows(a.segments, signal_key, a.batch_size):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 2:
            batch = batch[:, None, :]
        values, powers = extract_features(batch, a.fs, a.lowpass, a.eps)
        f[start:start + len(batch)] = values
        for i in range(len(batch)):
            qc.append(dict(row_index=start+i, n_nonfinite_signal_channels=int((~np.isfinite(batch[i]).all(axis=-1)).sum()),
                           n_zero_energy_channels=int((powers[i].sum(axis=-1) <= 0).sum()),
                           n_band_energies_below_log_floor=int((powers[i] < a.eps).sum())))
        print(f'  EEG {start + len(batch)}/{n}', end='\r', flush=True)
    print(flush=True)
    pd.DataFrame(qc).to_csv(out / 'spectral_qc.csv', index=False)
    e = np.empty((n, eshape[-1]), dtype=np.float64)
    print('Lecture et pooling des embeddings par lots...', flush=True)
    for start, batch in stream_rows(a.embeddings, embedding_key, a.batch_size):
        if batch.ndim > 2:
            pooled = batch.mean(axis=tuple(range(1, batch.ndim - 1)), dtype=np.float64)
        else:
            pooled = batch
        e[start:start + len(batch)] = pooled
        print(f'  Embeddings {start + len(batch)}/{n}', end='\r', flush=True)
    print(flush=True)
    # Sauvegarder les features de TOUS les essais, dans leur ordre d'origine.
    np.savez_compressed(out / 'spectral_features.npz', features=f,
                        labels=labels if labels is not None else np.array([]),
                        feature_names=np.asarray(FEATURE_NAMES), sampling_frequency=a.fs,
                        band_names=np.asarray([b[0] for b in BANDS]),
                        band_edges_hz=np.asarray([(low, min(high, a.lowpass, a.fs/2)) for _, low, high in BANDS]),
                        log_energy_floor=a.eps, fft_convention='abs(rfft(x))**2; no normalization/window/detrend',
                        row_index=np.arange(n))
    with (out / 'spectral_features.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['segment', 'channel', 'label'] + FEATURE_NAMES)
        for i in range(n):
            for ch in range(channels):
                writer.writerow([i, ch, labels[i] if labels is not None else '', *f[i, ch]])
    valid = np.isfinite(e).all(axis=1) & np.isfinite(f).all(axis=(1, 2))
    original = np.flatnonzero(valid)
    if len(original) < 14:
        raise ValueError('Au moins 14 essais finis requis pour garder >= 2 essais dans chaque partition')
    tr, rest = train_test_split(np.arange(len(original)), test_size=.30, random_state=a.seed)
    va, te = train_test_split(rest, test_size=.50, random_state=a.seed)
    split = np.full(n, 'excluded_nonfinite', dtype=object)
    for name, ids in [('train', tr), ('validation', va), ('test', te)]:
        split[original[ids]] = name
    pd.DataFrame({'row_index': np.arange(n), 'split': split}).to_csv(out / 'splits.csv', index=False)
    e, f_valid = e[valid], f[valid]
    # Même convention que le deuxième script : scalers appris sur train,
    # alpha choisi sur validation, modèle final réajusté sur train+validation.
    x_scaler = StandardScaler().fit(e[tr])
    xx = x_scaler.transform(e)
    trainval = np.concatenate([tr, va])
    results, channel_results = [], []
    final_predictions = np.full((len(te), channels, len(FEATURE_NAMES)), np.nan)
    print(f'Train={len(tr)}, validation={len(va)}, test={len(te)}, exclus={int((~valid).sum())}', flush=True)
    for j, name in enumerate(FEATURE_NAMES):
        y = f_valid[:, :, j]
        y_scaler = StandardScaler().fit(y[tr])
        yy = y_scaler.transform(y)
        choices = []
        for alpha in a.alphas:
            model = Ridge(alpha=alpha).fit(xx[tr], yy[tr])
            score = defined_mean(r2_per_channel(yy[va], prediction_2d(model, xx[va])))
            choices.append(score)
        if np.isfinite(choices).any():
            ai = int(np.argmax(np.where(np.isfinite(choices), choices, -np.inf)))
            status = 'ok'
        else:
            ai = int(np.argmax(a.alphas))
            status = 'undefined_validation_r2_fallback_max_alpha'
        alpha = a.alphas[ai]
        model = Ridge(alpha=alpha).fit(xx[trainval], yy[trainval])
        pred = y_scaler.inverse_transform(prediction_2d(model, xx[te]))
        final_predictions[:, :, j] = pred
        channel_r2 = r2_per_channel(y[te], pred)
        mean_r2 = defined_mean(channel_r2)
        results.append(dict(Feature=name, Best_Alpha=alpha, Validation_R2=choices[ai], Test_R2=mean_r2,
                            Defined_Test_Channels=int(np.isfinite(channel_r2).sum()), Status=status))
        for ch in range(channels):
            channel_results.append(dict(Feature=name, Channel=ch, Test_R2=channel_r2[ch],
                Test_RMSE=float(np.sqrt(np.mean((y[te, ch] - pred[:, ch]) ** 2))),
                Test_MAE=float(np.mean(np.abs(y[te, ch] - pred[:, ch]))),
                Train_Std=float(y[tr, ch].std()), TrainVal_Std=float(y[trainval, ch].std()),
                Test_Std=float(y[te, ch].std()),
                Low_Variance_Train=bool(y[tr, ch].std() <= 1e-8 + 1e-6 * abs(y[tr, ch].mean())),
                Low_Variance_Test=bool(y[te, ch].std() <= 1e-8 + 1e-6 * abs(y[te, ch].mean())), Undefined_Test_R2=not np.isfinite(channel_r2[ch])))
        print(f'{name:25s} alpha={alpha:g} val R²={choices[ai]:.4f} test R²={mean_r2:.4f}', flush=True)
    pd.DataFrame(results).to_csv(out / 'reve_spectral_ridge_results.csv', index=False)
    pd.DataFrame(channel_results).to_csv(out / 'reve_spectral_channel_results.csv', index=False)
    np.savez_compressed(out / 'test_predictions.npz', row_index=original[te],
                        y_true=f_valid[te], y_pred=final_predictions, feature_names=np.asarray(FEATURE_NAMES))
    config = vars(a) | dict(signal_key=signal_key, embedding_key=embedding_key, n_total=n, n_valid=len(original),
        n_channels=channels, split='random trial-level 70/15/15; subjects may overlap',
        alignment='original row order required; label equality checked if available',
        preprocessing='EEG unchanged; squared rfft, no detrend, no window, no normalization, natural logarithm; streaming mean pooling embeddings',
        bands=[(name, low, min(high, a.lowpass, a.fs/2)) for name, low, high in BANDS],
        log_floor='max(band_energy, eps); zero canonical energy produces NaN features',
        memory='numeric C-order arrays streamed from NPZ; only pooled embeddings and features retained',
        r2='raw, averaged over nonconstant test channels; constants NaN, never clipped',
        standardization='fit train only; final Ridge refit train+validation with frozen scalers', completed=True)
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nTerminé. Fichiers enregistrés dans : {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
