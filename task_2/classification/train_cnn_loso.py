"""Entraîner et sauvegarder un CNN REVE par sujet de test LOSO."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .cnn_reve import (cache_embeddings, choose_device, eligible_rows, fit_epoch, load_labels,
                       load_metadata, make_loader, make_model, metrics, predict, save_predictions, write_json, mean_pool_embeddings)

ROOT = Path(__file__).resolve().parents[1]


def select_epoch(array, labels, train, subjects, model_config, args, device, seed, folder):
    histories, choices, splits = [], [], []
    n_inner = min(args.inner_folds, len(np.unique(subjects[train])))
    for inner, (tr, va) in enumerate(GroupKFold(n_splits=n_inner).split(train, groups=subjects[train])):
        tr, va = train[tr], train[va]
        splits.extend(dict(Inner_Fold=inner, row_index=int(i), Subject=subjects[i], split=role)
                      for role, ids in [('train', tr), ('validation', va)] for i in ids)
        model = make_model(array, labels, tr, model_config, device, seed+inner)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loader = make_loader(array, labels, tr, args.batch_size, shuffle=True, seed=seed+inner)
        best_loss, best_epoch, stale = np.inf, 1, 0
        for epoch in range(1, args.epochs+1):
            loss = fit_epoch(model, loader, optimizer, device)
            score = metrics(labels[va], predict(model, array, labels, va, args.batch_size, device))
            histories.append(dict(Inner_Fold=inner, Epoch=epoch, Train_Loss=loss, **score))
            print(f'  validation {inner+1}/{n_inner}, epoch {epoch}/{args.epochs} : '
                  f'loss={score["Log_Loss"]:.4f}, macro-F1={score["Macro_F1"]:.3f}', flush=True)
            if score['Log_Loss'] < best_loss-1e-6:
                best_loss, best_epoch, stale = score['Log_Loss'], epoch, 0
            else:
                stale += 1
            if stale >= args.patience:
                break
        choices.append(best_epoch)
        del model, optimizer, loader
    # Fixed rule: median best epoch, then a fresh fit on ALL outer-training subjects.
    selected = max(1, int(np.floor(np.median(choices)+.5)))
    pd.DataFrame(histories).to_csv(folder/'inner_history.csv', index=False)
    pd.DataFrame(splits).to_csv(folder/'inner_splits.csv', index=False)
    return selected, choices


def main(model_type='cnn'):
    if model_type not in ('cnn', 'mlp'):
        raise ValueError('Type de modèle invalide')
    p = argparse.ArgumentParser(description=f'Entraîner un {model_type.upper()} sur les embeddings REVE en LOSO')
    data = ROOT.parent/'data'
    p.add_argument('--embeddings', type=Path, default=data/'reve_embeddings.npz')
    p.add_argument('--metadata', type=Path, default=data/'segments_metadata.csv')
    p.add_argument('--segments', type=Path, default=data/'segments_labeled.npz', help='Seuls les labels y sont lus')
    p.add_argument('--embedding-key', default='embeddings')
    p.add_argument('--label-key', default='y')
    p.add_argument('--subject-column', default='subject')
    p.add_argument('--exclude-subjects', nargs='*', default=[] if model_type == 'mlp' else ['sub-P02'])
    p.add_argument('--test-subjects', nargs='+', help='Option de vérification partielle, sinon tous les sujets retenus')
    p.add_argument('--out', type=Path, default=ROOT/'classification'/('results_mlp_tous_sujets' if model_type == 'mlp' else 'results_cnn_sans_P02'))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--inner-folds', type=int, default=3)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--width', type=int, default=64 if model_type == 'mlp' else 32)
    p.add_argument('--dropout', type=float, default=.25)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--resume', action='store_true', help='Reprendre les plis terminés du même --out')
    args = p.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.width, args.threads) < 1 or args.inner_folds < 2:
        p.error('Entiers positifs requis ; --inner-folds >= 2')
    if not 0 <= args.dropout < 1 or not np.isfinite([args.lr, args.weight_decay]).all() or args.lr <= 0 or args.weight_decay < 0:
        p.error('Paramètres numériques invalides')
    previous = None
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume or not (args.out/'config.json').is_file():
            p.error(f'{args.out} est non vide. Utiliser --resume pour cette expérience ou un autre --out.')
        previous = json.loads((args.out/'config.json').read_text(encoding='utf-8'))
    labels_raw = load_labels(args.embeddings, args.label_key)
    if not np.array_equal(labels_raw, load_labels(args.segments, args.label_key)):
        p.error('Labels des segments et des embeddings différents ou non alignés')
    metadata, subjects = load_metadata(args.metadata, len(labels_raw), args.subject_column)
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {key:str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if previous:
        if previous.get('model_type', 'cnn') != model_type:
            raise ValueError('Impossible de reprendre un CNN comme MLP, ou inversement')
        ignored = {'resume', 'device', 'threads', 'test_subjects'}
        changed = [k for k in vars(args) if k not in ignored and config[k] != previous.get(k)]
        if changed:
            raise ValueError(f'Paramètres différents de l’expérience à reprendre : {changed}')
    config.update(completed=False, model_type=model_type, protocol='outer LOSO; inner GroupKFold by subject; median best validation-loss epoch; fresh outer refit',
                  analysis='exploratory cohort excluding sub-P02' if 'sub-P02' in args.exclude_subjects else 'specified cohort',
                  normalization=('one mean/std per pooled dimension, fitted on training rows only' if model_type == 'mlp'
                                 else 'one mean/std per embedding dimension, fitted on training tokens only'),
                  torch_version=str(torch.__version__), device_used=str(device),
                  alignment='original row order required; labels EEG/embeddings and metadata length checked')
    if not previous:
        write_json(args.out/'config.json', config)
    print('Préparation du cache disque des embeddings (lecture par lots pour limiter la RAM)...', flush=True)
    array, provenance = cache_embeddings(args.embeddings, args.embedding_key, args.out/'cache')
    if len(array) != len(labels_raw):
        raise ValueError('Nombre d’embeddings incompatible avec les labels')
    rows, status = eligible_rows(array, subjects, args.exclude_subjects)
    source_shape = list(array.shape)
    groups = subjects[rows]
    included_subjects = np.unique(groups)
    if len(included_subjects) < 3:
        raise ValueError('Au moins trois sujets valides après exclusion sont nécessaires')
    classes, encoded = np.unique(labels_raw[rows], return_inverse=True)
    if len(classes) < 2:
        raise ValueError('Au moins deux classes sont nécessaires')
    labels = np.full(len(array), -1, dtype=np.int64)
    labels[rows] = encoded
    # Vocabulary only; class frequencies and any model statistics come from training.
    model_config = dict(input_shape=source_shape[1:], n_classes=len(classes), width=args.width, dropout=args.dropout)
    if model_type == 'mlp':
        model_config['model_type'] = 'mlp'
        pooled = mean_pool_embeddings(array, rows)
        array._mmap.close()
        array = pooled
        print(f'MLP : moyenne fixe canaux/fenêtres, forme d’entraînement {tuple(array.shape)}.', flush=True)
    pd.DataFrame({'row_index':np.arange(len(array)), 'Subject':subjects, 'status':status}).to_csv(args.out/'rows_qc.csv', index=False)
    pd.crosstab(pd.Series(subjects[rows], name='Subject'), pd.Series(labels_raw[rows], name='Label')).to_csv(args.out/'class_counts_by_subject.csv')
    class_names = []
    for label in classes:
        names = metadata.loc[labels_raw == label, 'trial_type'].unique().tolist() if 'trial_type' in metadata else []
        class_names.append(dict(label=label.item() if hasattr(label, 'item') else label,
                                description=names[0] if len(names)==1 else str(label)))
    requested = set(args.test_subjects or included_subjects.tolist())
    if not requested.issubset(set(included_subjects)):
        raise ValueError(f'Sujets de test absents ou exclus : {sorted(requested-set(included_subjects))}')
    config.update(input_shape=source_shape, training_input_shape=list(array.shape), model_config=model_config, classes=classes.tolist(),
                  class_names=class_names, included_subjects=included_subjects.tolist(),
                  source_embeddings=provenance, metadata_sha256=hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
                  labels_sha256=hashlib.sha256(labels_raw.dtype.str.encode()+labels_raw.tobytes()).hexdigest(),
                  n_rows_used=len(rows), n_rows_excluded=int(np.sum(status!='included')),
                  partial_run=requested != set(included_subjects))
    if previous and (previous.get('source_embeddings') != provenance or
                     previous.get('metadata_sha256') != config['metadata_sha256'] or
                     previous.get('classes') != config['classes'] or
                     previous.get('labels_sha256') != config['labels_sha256']):
        raise ValueError('Les données ne correspondent plus à l’expérience à reprendre')
    write_json(args.out/'config.json', config)
    print(f'{model_type.upper()} ; embeddings source : {tuple(source_shape)} ; classes : {classes.tolist()} ; '
          f'sujets retenus : {included_subjects.tolist()} ; exclus : {args.exclude_subjects}', flush=True)
    if config['partial_run']:
        print('Évaluation partielle : seuls les sujets demandés seront testés.', flush=True)
    scores, all_predictions, outer_splits = [], [], []
    for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(rows, groups=groups)):
        train, test = rows[tr], rows[te]
        subject = str(subjects[test[0]])
        if subject not in requested:
            continue
        folder = args.out/f'fold_{fold:02d}'
        print(f'\nLOSO {fold+1}/{len(included_subjects)} : test={subject}, train={len(train)}, test={len(test)}, '
              f'forme={tuple(array.shape[1:])}, appareil={device}', flush=True)
        roles = status.copy()
        roles[train], roles[test] = 'train', 'test'
        outer_splits.extend(dict(Fold=fold, Test_Subject=subject, row_index=i, Subject=str(s), split=str(role))
                            for i, (s, role) in enumerate(zip(subjects, roles)))
        if args.resume and (folder/'metrics.json').is_file() and (folder/'checkpoint.pt').is_file():
            score = json.loads((folder/'metrics.json').read_text())
            if score['Subject'] != subject:
                raise ValueError('Sujet du pli sauvegardé incohérent')
            scores.append(score)
            all_predictions.append(pd.read_csv(folder/'predictions.csv', dtype={'Subject':str}))
            print('  pli déjà terminé, conservé.', flush=True)
            continue
        if folder.exists():
            shutil.rmtree(folder)  # Incomplete generated fold; refit from its beginning.
        folder.mkdir()
        selected, inner_choices = select_epoch(array, labels, train, subjects, model_config, args, device,
                                               args.seed+1000*fold, folder)
        model = make_model(array, labels, train, model_config, device, args.seed+1000*fold+999)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loader = make_loader(array, labels, train, args.batch_size, shuffle=True, seed=args.seed+1000*fold+999)
        history = []
        for epoch in range(1, selected+1):
            loss = fit_epoch(model, loader, optimizer, device)
            history.append(dict(Epoch=epoch, Train_Loss=loss))
            print(f'  modèle final : epoch {epoch}/{selected}, loss={loss:.4f}', flush=True)
        pd.DataFrame(history).to_csv(folder/'train_history.csv', index=False)
        probability = predict(model, array, labels, test, args.batch_size, device)
        score = dict(Fold=fold, Subject=subject, N_Train=len(train), N_Test=len(test),
                     Selected_Epoch=selected, **metrics(labels[test], probability))
        majority = int(np.bincount(labels[train], minlength=len(classes)).argmax())
        majority_probability = np.zeros_like(probability)
        majority_probability[:, majority] = 1.
        reference = metrics(labels[test], majority_probability)
        score.update(Majority_Accuracy=reference['Accuracy'], Majority_Macro_F1=reference['Macro_F1'])
        scores.append(score)
        frame = save_predictions(folder, labels, probability, test, subjects, classes)
        frame.insert(0, 'Fold', fold)
        all_predictions.append(frame)
        checkpoint = dict(format_version=1, model_config=model_config,
            model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
            classes=classes.tolist(), test_subject=subject, train_subjects=np.unique(subjects[train]).tolist(),
            test_rows=test.tolist(), train_rows=train.tolist(), excluded_subjects=args.exclude_subjects,
            test_labels=labels_raw[test].tolist(), selected_epoch=selected, inner_best_epochs=inner_choices,
            source_shape=source_shape, embedding_key=args.embedding_key, label_key=args.label_key,
            seed=args.seed+1000*fold+999, torch_version=str(torch.__version__), device_used=str(device),
            subject_column=args.subject_column, metadata_sha256=config['metadata_sha256'])
        torch.save(checkpoint, folder/'checkpoint.pt.tmp')
        (folder/'checkpoint.pt.tmp').replace(folder/'checkpoint.pt')
        write_json(folder/'metrics.json', score)
        pd.DataFrame(scores).to_csv(args.out/'fold_metrics.csv', index=False)
        pd.concat(all_predictions, ignore_index=True).to_csv(args.out/'predictions.csv', index=False)
        pd.DataFrame(outer_splits).to_csv(args.out/'outer_splits.csv', index=False)
        print(f'  Test {subject} : accuracy={score["Accuracy"]:.3f}, macro-F1={score["Macro_F1"]:.3f}', flush=True)
        del model, optimizer, loader
    result = pd.DataFrame(scores)
    pd.DataFrame(scores).to_csv(args.out/'fold_metrics.csv', index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(args.out/'predictions.csv', index=False)
    pd.DataFrame(outer_splits).to_csv(args.out/'outer_splits.csv', index=False)
    columns = ['Accuracy', 'Balanced_Accuracy', 'Macro_F1', 'Log_Loss', 'Majority_Accuracy', 'Majority_Macro_F1']
    result[columns].agg(['mean','std']).to_csv(args.out/'metrics_summary.csv')
    config.update(completed=True, n_folds_completed=len(result))
    # The NPY cache is reproducible; checkpoints need no cached embeddings.
    if hasattr(array, '_mmap'):
        array._mmap.close()
    del array
    shutil.rmtree(args.out/'cache')
    write_json(args.out/'config.json', config)
    print(f'\nTerminé. Modèles et résultats : {args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
