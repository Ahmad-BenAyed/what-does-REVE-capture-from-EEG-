"""Lightweight CNN, training-only normalization and NPZ input helpers."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


class EmbeddingCNN(nn.Module):
    """Input N x electrodes x temporal patches x embedding dimensions.

    Convolutions mix embedding dimensions and neighboring temporal patches.
    They do not assume adjacency between consecutive electrode indices.
    """
    def __init__(self, input_shape, n_classes, width=32, dropout=.25):
        super().__init__()
        self.input_shape = tuple(int(v) for v in input_shape)
        electrodes, _, dimensions = self.input_shape
        self.register_buffer('input_mean', torch.zeros(dimensions))
        self.register_buffer('input_std', torch.ones(dimensions))
        self.features = nn.Sequential(
            nn.Conv2d(dimensions, width, kernel_size=1), nn.GELU(),
            nn.Conv2d(width, width, kernel_size=(1, 3), padding=(0, 1), groups=width),
            nn.GELU(), nn.Conv2d(width, width, kernel_size=1), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width * electrodes, 64),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, embeddings):
        if tuple(embeddings.shape[1:]) != self.input_shape:
            raise ValueError(f'Entrée attendue N×{self.input_shape}, reçue {tuple(embeddings.shape)}')
        # Erased embeddings must enter here, BEFORE the saved normalization.
        x = (embeddings - self.input_mean) / self.input_std
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.features(x).mean(dim=-1)
        return self.head(x)


class EmbeddingMLP(nn.Module):
    """Global mean pooling -> train-only scaling -> D -> width -> K."""
    def __init__(self, input_shape, n_classes, width=64, dropout=.25):
        super().__init__()
        self.input_shape = tuple(int(v) for v in input_shape)
        dimensions = self.input_shape[-1]
        self.register_buffer('input_mean', torch.zeros(dimensions))
        self.register_buffer('input_std', torch.ones(dimensions))
        self.head = nn.Sequential(nn.Linear(dimensions, width), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(width, n_classes))

    def forward(self, embeddings):
        # Training uses pre-pooled vectors; frozen replay accepts the original tensor.
        shape = tuple(embeddings.shape[1:])
        if embeddings.ndim != 4 or shape not in (self.input_shape, (1, 1, self.input_shape[-1])):
            raise ValueError(f'Entrée MLP incompatible : {tuple(embeddings.shape)}')
        x = embeddings.mean(dim=(1, 2))
        return self.head((x-self.input_mean)/self.input_std)


def build_classifier(config):
    options = dict(config)
    kind = options.pop('model_type', 'cnn')  # Existing CNN checkpoints remain compatible.
    if kind not in ('cnn', 'mlp'):
        raise ValueError(f'Type de modèle inconnu : {kind}')
    return (EmbeddingMLP if kind == 'mlp' else EmbeddingCNN)(**options)


def mean_pool_embeddings(array, indices=None, batch_size=16):
    """Fixed pooling, no learned statistics. Keep original row indices."""
    indices = np.arange(len(array)) if indices is None else np.asarray(indices)
    pooled = np.zeros((len(array), 1, 1, array.shape[-1]), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start+batch_size]
        block = np.asarray(array[rows], dtype=np.float32)
        pooled[rows, 0, 0] = block.mean(axis=(1, 2), dtype=np.float32)
    return pooled


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(value):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    if value == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA indisponible. Utiliser --device cpu ou installer PyTorch pour votre GPU.')
    return torch.device(value)


def canonical_array(array):
    if array.ndim == 2:
        return array[:, None, None, :]
    if array.ndim == 3:
        return array[:, :, None, :]
    if array.ndim != 4:
        raise ValueError(f'Embeddings attendus N×C×P×D (ou N×D / N×C×D), reçus {array.shape}')
    return array


def cache_embeddings(path, key, cache):
    """Extract one numeric NPY member to disk; avoid a full in-memory NPZ copy."""
    path, cache = Path(path), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    member = key + '.npy'
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise ValueError(f'Clé embeddings absente : {key}')
        info = archive.getinfo(member)
        provenance = dict(path=str(path.resolve()), file_size=path.stat().st_size,
                          member=member, member_size=info.file_size, member_crc=int(info.CRC))
        tag = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:20]
        target = cache / f'embeddings_{tag}.npy'
        if not target.exists() or target.stat().st_size != info.file_size:
            partial = target.with_suffix('.partial')
            try:
                with archive.open(member) as source, partial.open('wb') as handle:
                    shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
    array = np.load(target, mmap_mode='r', allow_pickle=False)
    try:
        if array.dtype.kind not in 'iuf' or not all(array.shape):
            raise ValueError('Les embeddings doivent être un tableau numérique non vide')
        array = canonical_array(array)
        return array, provenance
    except Exception:
        array._mmap.close()
        raise


def load_labels(path, key='y'):
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise ValueError(f'Labels absents : clé {key!r} dans {path}')
        y = np.asarray(data[key])
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or len(y) == 0:
        raise ValueError('Labels attendus : vecteur de longueur N')
    if y.dtype.kind in 'iuf' and not np.isfinite(y).all():
        raise ValueError('Labels non finis')
    if y.dtype.kind in 'US' and np.any(np.char.strip(y.astype(str)) == ''):
        raise ValueError('Labels vides')
    return y


def load_metadata(path, n, column='subject'):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(frame) != n or column not in frame:
        raise ValueError('Metadata : une ligne par embedding et une colonne sujet sont requises')
    subjects = frame[column].str.strip().to_numpy(str)
    if np.isin(np.char.lower(subjects), ['', 'unknown', 'none', 'nan', 'null']).any():
        raise ValueError('Identifiants sujets manquants ou inconnus')
    return frame, subjects


def eligible_rows(array, subjects, excluded, batch_size=16):
    missing = set(excluded) - set(subjects)
    if missing:
        raise ValueError(f'Sujets à exclure absents : {sorted(missing)}')
    included = ~np.isin(subjects, excluded)
    finite = np.zeros(len(array), dtype=bool)
    for start in range(0, len(array), batch_size):
        block = np.asarray(array[start:start+batch_size])
        finite[start:start+len(block)] = np.isfinite(block).all(axis=(1, 2, 3))
    status = np.full(len(array), 'included', dtype=object)
    status[~finite] = 'excluded_nonfinite'
    status[~included] = 'excluded_subject'
    return np.flatnonzero(included & finite), status


def fit_normalization(array, indices, batch_size=16, pool_tokens=False):
    """Welford merge of token statistics from TRAINING rows only."""
    count = 0
    mean = np.zeros(array.shape[-1], dtype=np.float64)
    m2 = mean.copy()
    for start in range(0, len(indices), batch_size):
        block = np.asarray(array[indices[start:start+batch_size]], dtype=np.float64)
        if pool_tokens:
            block = block.mean(axis=(1, 2), keepdims=True)
        block = block.reshape(-1, array.shape[-1])
        if not np.isfinite(block).all():
            raise ValueError('Embedding non fini dans les lignes d’entraînement')
        batch_count = len(block)
        batch_mean = block.mean(axis=0)
        batch_m2 = ((block-batch_mean)**2).sum(axis=0)
        total = count+batch_count
        delta = batch_mean-mean
        mean += delta*batch_count/total
        m2 += batch_m2+delta**2*count*batch_count/total
        count = total
    if not count:
        raise ValueError('Entraînement vide')
    std = np.sqrt(np.maximum(m2/count, 0))
    std[std < 1e-8] = 1.
    return mean.astype(np.float32), std.astype(np.float32)


class RowsDataset(Dataset):
    def __init__(self, array, labels, indices):
        self.array, self.labels, self.indices = array, labels, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row = int(self.indices[item])
        return torch.from_numpy(np.array(self.array[row], dtype=np.float32, copy=True)), int(self.labels[row])


def make_loader(array, labels, indices, batch_size, shuffle=False, seed=42):
    return DataLoader(RowsDataset(array, labels, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def make_model(array, labels, train, config, device, seed):
    seed_everything(seed)
    model = build_classifier(config).to(device)
    mean, std = fit_normalization(array, train, pool_tokens=config.get('model_type') == 'mlp')
    model.input_mean.copy_(torch.as_tensor(mean, device=device))
    model.input_std.copy_(torch.as_tensor(std, device=device))
    classes = np.unique(labels[train])
    if len(classes) != config['n_classes']:
        raise ValueError('Une classe est absente des sujets d’entraînement de ce pli')
    return model


def fit_epoch(model, loader, optimizer, device):
    model.train()
    total, count = 0., 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise ValueError('Perte non finie pendant l’entraînement')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
        optimizer.step()
        total += float(loss.detach())*len(y)
        count += len(y)
    return total/count


def predict(model, array, labels, indices, batch_size, device):
    model.eval()
    parts = []
    with torch.inference_mode():
        for x, _ in make_loader(array, labels, indices, batch_size):
            parts.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    probabilities = np.concatenate(parts)
    if not np.isfinite(probabilities).all():
        raise ValueError('Prédictions non finies')
    return probabilities


def metrics(y, probabilities):
    labels = np.arange(probabilities.shape[1])
    prediction = probabilities.argmax(axis=1)
    present = np.unique(y)
    return dict(Accuracy=float(accuracy_score(y, prediction)),
                Balanced_Accuracy=float(recall_score(y, prediction, labels=present, average='macro', zero_division=0)),
                Macro_F1=float(f1_score(y, prediction, labels=labels, average='macro', zero_division=0)),
                Log_Loss=float(log_loss(y, probabilities, labels=labels)))


def save_predictions(out, y, probabilities, rows, subjects, classes):
    prediction = probabilities.argmax(axis=1)
    frame = pd.DataFrame({'row_index':rows, 'Subject':subjects[rows],
                          'True_Label':np.asarray(classes)[y[rows]],
                          'Predicted_Label':np.asarray(classes)[prediction]})
    for i in range(len(classes)):
        frame[f'Probability_Class_{i}'] = probabilities[:, i]
    frame.to_csv(Path(out)/'predictions.csv', index=False)
    matrix = confusion_matrix(y[rows], prediction, labels=np.arange(len(classes)))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(Path(out)/'confusion_matrix.csv')
    return frame


def load_classifier(path, device='cpu'):
    # Only load checkpoints you created or otherwise trust.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 1:
        raise ValueError('Format de checkpoint inconnu')
    model = build_classifier(payload['model_config'])
    model.load_state_dict(payload['model_state'])
    model.to(device).eval().requires_grad_(False)
    return model, payload
