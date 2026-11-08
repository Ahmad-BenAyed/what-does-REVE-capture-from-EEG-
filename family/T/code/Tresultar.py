"""Extraction des 10 features temporelles + régression Ridge depuis REVE.
Exemple, depuis le dossier data :
python 05_temporal_extract_and_regress.py --out results_temporal
Attention : découpage aléatoire par essai, pas une évaluation indépendante par sujet.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    'T001_Activity', 'T002_Mobility', 'T003_Complexity', 'T004_Std',
    'T005_RMS', 'T006_Kurtosis', 'T007_ZeroCrossing', 'T008_LineLength',
    'T009_DerivativeStd', 'T010_PeakToPeak',
]


def extract_features(x, fs=200):
    # Mêmes définitions et même epsilon que votre code d'extraction.
    dx = np.diff(x, axis=-1) * fs
    ddx = np.diff(dx, axis=-1) * fs
    vx, vd, vdd = np.var(x, axis=-1), np.var(dx, axis=-1), np.var(ddx, axis=-1)
    eps = 1e-12
    mobility = np.sqrt(vd / (vx + eps))
    centered = x - np.mean(x, axis=-1, keepdims=True)
    return np.stack([
        vx, mobility, np.sqrt(vdd / (vd + eps)) / (mobility + eps),
        np.sqrt(vx), np.sqrt(np.mean(x ** 2, axis=-1)),
        np.mean(centered ** 4, axis=-1) / (vx ** 2 + eps),
        np.mean(x[..., :-1] * x[..., 1:] < 0, axis=-1),
        np.mean(np.abs(np.diff(x, axis=-1)), axis=-1),
        np.sqrt(vd), np.ptp(x, axis=-1),
    ], axis=-1)


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


def choose_key(data, explicit, kind):
    if explicit:
        if explicit not in data.files:
            raise ValueError(f'Clé {explicit!r} absente. Clés : {data.files}')
        return explicit
    if kind == 'eeg':
        candidates = [k for k in data.files if k.lower() in ('segments', 'x', 'eeg', 'signals', 'data')
                      and data[k].ndim in (2, 3) and data[k].dtype.kind in 'iuf']
    elif 'embeddings' in data.files:
        return 'embeddings'
    else:
        candidates = [k for k in data.files if data[k].ndim >= 2 and data[k].dtype.kind in 'iuf']
    if len(candidates) != 1:
        raise ValueError(f'Clé {kind} ambiguë : {candidates}. Utiliser --signal-key ou --embedding-key.')
    return candidates[0]


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
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--alphas', nargs='+', type=float, default=[.1, 1., 10., 100.])
    p.add_argument('--out', default='results_temporal')
    a = p.parse_args()
    if not np.isfinite(a.fs) or a.fs <= 0 or any(not np.isfinite(v) or v <= 0 for v in a.alphas):
        p.error('fs et alphas doivent être finis et strictement positifs')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'{out} contient déjà des fichiers. Choisissez un autre --out.')
    # N'utiliser que vos propres NPZ de confiance (allow_pickle pour anciens exports).
    with np.load(a.segments, allow_pickle=True) as data:
        signal_key = choose_key(data, a.signal_key, 'eeg')
        x = np.asarray(data[signal_key], dtype=np.float64)
        labels = load_labels(data)
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3 or min(x.shape) == 0 or x.shape[-1] < 3:
        raise ValueError('EEG attendu : N×C×T, avec au moins trois points par essai')
    n, channels, time = x.shape
    if labels is not None and len(labels) != n:
        raise ValueError('Nombre de labels différent du nombre de segments')
    with np.load(a.embeddings, allow_pickle=True) as data:
        embedding_key = choose_key(data, a.embedding_key, 'embedding')
        e = np.asarray(data[embedding_key], dtype=np.float64)
        embedding_labels = load_labels(data)
    if e.ndim > 2:
        e = e.mean(axis=tuple(range(1, e.ndim - 1)))
    if e.ndim != 2 or len(e) != n:
        raise ValueError(f'Embeddings incompatibles : {e.shape}, EEG : {x.shape}')
    if embedding_labels is not None and len(embedding_labels) != n:
        raise ValueError('Nombre de labels embeddings différent du nombre de segments')
    if labels is not None and embedding_labels is not None and not np.array_equal(labels, embedding_labels):
        raise ValueError('Labels EEG/embeddings non alignés')
    print(f'EEG : {x.shape}, fs={a.fs} Hz, durée={time/a.fs:g} s ; embeddings poolés : {e.shape}', flush=True)
    print('Extraction des dix features temporelles...', flush=True)
    f = extract_features(x, a.fs)
    # Sauvegarder les features de TOUS les essais, dans leur ordre d'origine.
    np.savez_compressed(out / 'temporal_features.npz', features=f,
                        labels=labels if labels is not None else np.array([]),
                        feature_names=np.asarray(FEATURE_NAMES), sampling_frequency=a.fs,
                        row_index=np.arange(n))
    with (out / 'temporal_features.csv').open('w', newline='', encoding='utf-8') as handle:
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
    final_predictions = np.full((len(te), channels, 10), np.nan)
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
                Test_Std=float(y[te, ch].std()), Undefined_Test_R2=not np.isfinite(channel_r2[ch])))
        print(f'{name:25s} alpha={alpha:g} val R²={choices[ai]:.4f} test R²={mean_r2:.4f}', flush=True)
    pd.DataFrame(results).to_csv(out / 'reve_temporal_ridge_results.csv', index=False)
    pd.DataFrame(channel_results).to_csv(out / 'reve_temporal_channel_results.csv', index=False)
    np.savez_compressed(out / 'test_predictions.npz', row_index=original[te],
                        y_true=f_valid[te], y_pred=final_predictions, feature_names=np.asarray(FEATURE_NAMES))
    config = vars(a) | dict(signal_key=signal_key, embedding_key=embedding_key, n_total=n, n_valid=len(original),
        n_channels=channels, split='random trial-level 70/15/15; subjects may overlap',
        alignment='original row order required; label equality checked if available',
        preprocessing='EEG unchanged; feature formulas from supplied code; mean pooling of embedding intermediate axes',
        r2='raw, averaged over nonconstant test channels; constants NaN, never clipped',
        standardization='fit train only; final Ridge refit train+validation with frozen scalers', completed=True)
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nTerminé. Fichiers enregistrés dans : {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
