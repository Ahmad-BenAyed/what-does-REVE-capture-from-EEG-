"""Réévaluer un CNN ou MLP sauvegardé, sans ajuster ses poids ou sa normalisation."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from ..paths import Paths, output_problem, prepare_output

import numpy as np
import pandas as pd
import torch

from .cnn_reve import (cache_embeddings, choose_device, load_classifier, load_labels,
                       load_metadata, metrics, predict, save_predictions, mean_pool_embeddings)


def main():
    paths = Paths()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--embeddings', type=Path, default=paths.input('embeddings'))
    p.add_argument('--metadata', type=Path, default=paths.input('metadata'))
    p.add_argument('--reference', type=Path, help='Compare to the original fold predictions.csv')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if min(a.batch_size, a.threads) < 1:
        p.error('batch-size et threads doivent être positifs')
    if output_problem(a.out):
        p.error('Le dossier de sortie doit être vide')
    torch.set_num_threads(a.threads)
    device = choose_device(a.device)
    model, checkpoint = load_classifier(a.checkpoint, device)
    labels_raw = load_labels(a.embeddings, checkpoint['label_key'])
    _, subjects = load_metadata(a.metadata, len(labels_raw), checkpoint['subject_column'])
    if hashlib.sha256(a.metadata.read_bytes()).hexdigest() != checkpoint['metadata_sha256']:
        raise ValueError('Utiliser le même CSV metadata, dans le même ordre, que pendant l’entraînement')
    rows = np.asarray(checkpoint['test_rows'])
    if len(labels_raw) != checkpoint['source_shape'][0]:
        raise ValueError('Le nouveau NPZ doit conserver toutes les lignes dans leur ordre original')
    if not np.array_equal(labels_raw[rows], checkpoint['test_labels']):
        raise ValueError('Les labels des lignes de test ont changé')
    if not np.all(subjects[rows] == checkpoint['test_subject']):
        raise ValueError('Les lignes de test ne correspondent plus au sujet sauvegardé')
    if set(subjects[rows]) & set(checkpoint['train_subjects']):
        raise ValueError('Chevauchement sujets train/test dans le checkpoint')
    classes = checkpoint['classes']
    lookup = {v:i for i,v in enumerate(classes)}
    labels = np.full(len(labels_raw), -1, dtype=np.int64)
    labels[rows] = [lookup[v] for v in labels_raw[rows].tolist()]
    reference = None
    if a.reference:
        saved = pd.read_csv(a.reference, dtype={"Subject": str}).set_index("row_index")
        cols = [f"Probability_Class_{i}" for i in range(len(classes))]
        if (not saved.index.is_unique or set(saved.index) != set(rows)
                or not (saved.loc[rows, "Subject"] == checkpoint["test_subject"]).all()
                or not np.array_equal(saved.loc[rows, "True_Label"].astype(str), labels_raw[rows].astype(str))):
            p.error("Reference predictions do not match this fold")
        reference = saved.loc[rows, cols].to_numpy(dtype=np.float32)
        if not np.isfinite(reference).all() or (reference < 0).any() or not np.allclose(reference.sum(axis=1), 1):
            p.error("Invalid reference probabilities")
    prepare_output(a.out)
    with tempfile.TemporaryDirectory(prefix='cnn_eval_', dir=a.out) as cache:
        array, _ = cache_embeddings(a.embeddings, checkpoint['embedding_key'], cache)
        try:
            expected = checkpoint['source_shape']
            pooled_shape = [expected[0], 1, 1, expected[-1]]
            pooled_mlp = checkpoint['model_config'].get('model_type') == 'mlp' and list(array.shape) == pooled_shape
            if list(array.shape) != expected and not pooled_mlp:
                raise ValueError('Forme incompatible avec le modèle sauvegardé')
            for start in range(0, len(rows), a.batch_size):
                if not np.isfinite(array[rows[start:start+a.batch_size]]).all():
                    raise ValueError('Les nouveaux embeddings de test contiennent des valeurs non finies')
            scoring_array = (mean_pool_embeddings(array, rows)
                             if checkpoint['model_config'].get('model_type') == 'mlp' else array)
            probabilities = predict(model, scoring_array, labels, rows, a.batch_size, device)
            del scoring_array
        finally:
            array._mmap.close()
            del array
    score = metrics(labels[rows], probabilities)
    save_predictions(a.out, labels, probabilities, rows, subjects, classes)
    if reference is not None:
        baseline = metrics(labels[rows], reference)
        score.update({f"Reference_{key}": value for key, value in baseline.items()})
        score.update({f"Delta_{key}": score[key] - value for key, value in baseline.items()})
        score["Max_absolute_probability_change"] = float(np.max(np.abs(probabilities - reference)))
        score["Reference_probabilities_match"] = bool(np.allclose(probabilities, reference, atol=1e-5))
    score.update(Subject=checkpoint['test_subject'], checkpoint=str(a.checkpoint.resolve()),
                 embeddings=str(a.embeddings.resolve()), frozen_weights=True, frozen_normalization=True)
    (a.out/'metrics.json').write_text(json.dumps(score, indent=2), encoding='utf-8')
    print(pd.Series(score).to_string())


if __name__ == '__main__':
    main()
