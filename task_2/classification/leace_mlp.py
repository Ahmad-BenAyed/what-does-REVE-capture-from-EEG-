"""LEACE par caractéristique EEG, sur les entrées moyennées du MLP LOSO."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .cnn_reve import (cache_embeddings, load_classifier, load_labels, load_metadata,
                       mean_pool_embeddings, metrics)

FAMILIES = {'T': ('temporal', 10), 'F': ('spectral', 16), 'TF': ('tf', 11),
            'C': ('c', 7), 'R': ('r', 14), 'X': ('x', 5)}


def read_targets(root, subjects, required):
    targets = {}
    for family, (suffix, count) in FAMILIES.items():
        path = root / family / 'resultats' / ('results_' + suffix) / 'test_predictions.npz'
        with np.load(path, allow_pickle=False) as data:
            rows = data['row_index']
            if (rows.dtype.kind not in 'iu' or len(np.unique(rows)) != len(rows)
                    or np.any(rows < 0) or np.any(rows >= len(subjects))
                    or not np.isin(required, rows).all()
                    or not np.array_equal(subjects[rows], data['subject'].astype(str))):
                raise ValueError(f'{path}: alignement incomplet avec les dix plis MLP')
            names = [s.rsplit(':', 1)[0] for s in data['column_names'].astype(str)]
            values = data['y_true']
            if values.shape != (len(rows), len(names)) or not np.isfinite(values).all():
                raise ValueError(f'{path}: cibles invalides')
            features = list(dict.fromkeys(names))
            if len(features) != count or set(features) & targets.keys():
                raise ValueError(f'{path}: vérifier le catalogue de caractéristiques')
            for feature in features:
                mask = np.array(names) == feature
                z = np.full((len(subjects), mask.sum()), np.nan)
                z[rows] = values[:, mask]
                targets[feature] = z
    return targets


def retained(s, shape):
    return s > np.finfo(float).eps * max(shape) * s[0]


def probabilities(model, x):
    with torch.inference_mode():
        tensor = torch.from_numpy(x.astype(np.float32))[:, None, None, :]
        return model(tensor).softmax(dim=1).numpy()


def run(args):
    config = json.loads((args.models / 'config.json').read_text(encoding='utf-8'))
    meta = args.data / 'segments_metadata.csv'
    path = args.data / 'reve_embeddings.npz'
    labels = load_labels(path)
    _, subjects = load_metadata(meta, len(labels))
    digest = hashlib.sha256(meta.read_bytes()).hexdigest()
    if (digest != config['metadata_sha256'] or
            hashlib.sha256(labels.dtype.str.encode()+labels.tobytes()).hexdigest() != config['labels_sha256']):
        raise ValueError('Métadonnées ou étiquettes différentes du MLP')
    paths = sorted(args.models.glob('fold_*/checkpoint.pt'))
    folds = [load_classifier(p) for p in paths]
    expected = set(np.unique(subjects))
    if len(paths) != 10 or len(expected) != 10 or {c['test_subject'] for _, c in folds} != expected:
        raise ValueError('Utiliser les dix checkpoints du MLP tous sujets')
    required = np.concatenate([c['test_rows'] for _, c in folds])
    if len(np.unique(required)) != len(required) or min(required) < 0 or max(required) >= len(labels):
        raise ValueError('Lignes de test invalides')
    targets = read_targets(args.family, subjects, required)
    array, source = cache_embeddings(path, 'embeddings', args.models / 'cache')
    if source['member_crc'] != config['source_embeddings']['member_crc'] or list(array.shape) != config['input_shape']:
        raise ValueError('Embeddings différents de ceux du MLP')
    x = mean_pool_embeddings(array, required)[:, 0, 0].astype(float)
    del array
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Choisir un dossier --out vide')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'config.json').write_text(json.dumps(dict(seed=args.seed, controls=args.controls, metadata_sha256=digest, embeddings=source, models=str(args.models.resolve()), features=list(targets)), indent=2), encoding='utf-8')
    records = []
    for fold, ((model, c), checkpoint) in enumerate(zip(folds, paths)):
        subject = c['test_subject']
        print(f'{subject} : 63 effacements', flush=True)
        train, test = np.asarray(c['train_rows']), np.asarray(c['test_rows'])
        if (c['model_config'].get('model_type') != 'mlp' or c['metadata_sha256'] != digest
                or not np.array_equal(np.sort(train), np.sort(required[subjects[required] != subject]))
                or not np.all(subjects[test] == subject) or not np.array_equal(labels[test], c['test_labels'])):
            raise ValueError(f'{subject}: checkpoint ou séparation LOSO incompatible')
        y = np.array([c['classes'].index(v) for v in labels[test]])
        base = probabilities(model, x[test])
        saved = pd.read_csv(checkpoint.parent / 'predictions.csv').set_index('row_index').loc[test]
        if not np.allclose(base, saved.filter(regex='^Probability_Class_').values, atol=1e-5):
            raise ValueError(f'{subject}: reproduction des probabilités initiales impossible')
        baseline = metrics(y, base)
        folder = args.out / subject
        folder.mkdir()
        mean = x[train].mean(0)
        centered = x[train] - mean
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        keep = retained(s, centered.shape)
        u, s, vt = u[:, keep], s[keep], vt[keep]
        white_test = ((x[test]-mean) @ vt.T) / s
        rng = np.random.default_rng(args.seed + fold)
        np.savez(folder / 'fold.npz', mean=mean, train_rows=train, test_rows=test,
                 classes=c['classes'], labels=labels[test], original_probabilities=base)
        for feature, values in targets.items():
            z = values[train] - values[train].mean(0)
            scale = z.std(0)
            z[:, np.ptp(values[train], axis=0) == 0] = 0
            z /= np.where(scale > 0, scale, 1)
            cross = u.T @ z
            basis, sv, _ = np.linalg.svd(cross, full_matrices=False)
            basis = basis[:, retained(sv, cross.shape)]
            rank = basis.shape[1]
            # Row-vector LEACE: whiten, project, unwhiten, restore the mean.
            left = (vt.T / s) @ basis
            right = basis.T @ (s[:, None] * vt)
            delta = ((x[test] - mean) @ left) @ right
            erased = x[test] - delta
            clean_train = centered - (centered @ left) @ right
            residual = np.linalg.norm(clean_train.T @ z)
            fraction = residual / max(np.linalg.norm(cross * s[:, None]), np.finfo(float).tiny)
            if fraction > 1e-7:
                raise ArithmeticError(f'{subject}/{feature}: covariance résiduelle {fraction}')
            # Compare squared error to a fixed training-mean predictor on the held subject.
            target_test = (values[test] - values[train].mean(0)) / np.where(scale > 0, scale, 1)
            active = np.ptp(values[train], axis=0) > 0
            def probe_skill(a, b):
                if not active.any():
                    return np.nan
                target = target_test[:, active]
                denom = np.sum(target**2)
                if denom == 0:
                    return np.nan
                probe = Ridge(alpha=1., fit_intercept=False, solver='cholesky').fit(a, z[:, active])
                pred = probe.predict(b).reshape(target.shape)
                return 1 - np.sum((target-pred)**2) / denom
            before = probe_skill(centered, x[test]-mean)
            after = probe_skill(clean_train, erased-mean)
            p = probabilities(model, erased)
            np.savez_compressed(folder / (feature + '.npz'), left=left, right=right,
                                erased_probabilities=p, **({'embeddings': erased.astype('f4')} if args.save_embeddings else {}))
            for repetition in range(args.controls + 1):
                if repetition:
                    random_basis = np.linalg.qr(rng.normal(size=(len(s), rank)), mode='reduced')[0]
                    change = (white_test @ random_basis) @ (random_basis.T @ (s[:, None]*vt))
                    prob = probabilities(model, x[test]-change)
                else:
                    change, prob = delta, p
                scores = metrics(y, prob)
                record = dict(Subject=subject, Feature=feature, Control=repetition, Rank=rank,
                              Train_cov_residual=fraction, Probe_skill_original=before, Probe_skill_erased=after, Test_RMS_change=np.sqrt(np.mean(change**2)))
                for metric, value in scores.items():
                    record[metric] = value
                    record['Original_'+metric] = baseline[metric]
                    record['Delta_'+metric] = value - baseline[metric]
                records.append(record)
        pd.DataFrame(records).to_csv(args.out / 'by_subject.csv', index=False)
    frame = pd.DataFrame(records)
    columns = ['Delta_Accuracy','Delta_Balanced_Accuracy','Delta_Macro_F1','Delta_Log_Loss','Test_RMS_change']
    actual = frame[frame.Control == 0].groupby('Feature')[columns].mean()
    controls = frame[frame.Control > 0].groupby(['Feature','Subject'])[columns].mean().groupby('Feature').mean()
    actual.join(controls.add_prefix('Random_')).to_csv(args.out / 'summary.csv')
    print(args.out / 'summary.csv')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root.parent / 'data')
    parser.add_argument('--family', type=Path, default=root / 'family')
    parser.add_argument('--models', type=Path, default=root / 'classification/results_mlp_tous_sujets')
    parser.add_argument('--out', type=Path, default=root / 'classification/results_leace_mlp')
    parser.add_argument('--controls', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-embeddings', action='store_true')
    args = parser.parse_args()
    if args.controls < 1:
        parser.error('Au moins un contrôle aléatoire requis')
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        run(args)
