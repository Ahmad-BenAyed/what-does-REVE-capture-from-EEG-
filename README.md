# What does REVE capture from EEG?

Exploration of the representations learned by **REVE** (`brain-bzh/reve-base`), a pretrained EEG foundation model, on a 3-class mental-task EEG dataset. The project extracts REVE embeddings from EEG trials, then probes them to see whether they separate the mental tasks (PCA visualisation, linear probe, small CNN).

## Task and data

EEG recorded during three mental tasks (BIDS format, dataset id `nm000290`, BCI Graz-style acquisition):

| Label | Class | Description |
|-------|-------|-------------|
| 0 | `mi` | Motor imagery |
| 1 | `sub` | Mental subtraction |
| 2 | `word` | Word generation |

- 41 EEG channels (order is hard-coded in the embedding scripts)
- Trials are cut using the `onset` / `duration` columns of each `*_events.tsv`



## Pipeline

```
raw EDF + events.tsv
      │  prepare_all_data.py
      ▼
data/segments_labeled.npz   (X: trials × channels × time, y)
data/segments_metadata.csv
      │  embeddings_all_data.py   (REVE)
      ▼
data/reve_embeddings.npz    (embeddings: trials × channels × time-patches × dim, y)
      │
      ├── analyze_embeddings.py  → PCA plot + logistic-regression probe
      └── CNN.py                 → small CNN trained on the embeddings
```

### Preprocessing (`prepare_all_data.py`)
1. Read each EDF file with MNE and resample to **200 Hz** (REVE's expected rate).
2. Band-pass filter **0.5–99.5 Hz**.
3. Z-score each channel per file, then clip to **±15 σ**.
4. Extract one segment per `mi` / `sub` / `word` event; skip invalid or out-of-range events and segments whose length differs from the first one (needed to stack into one array).

### Embedding extraction (`embeddings_all_data.py`)
- Loads `brain-bzh/reve-base` and the electrode position bank `brain-bzh/reve-positions` from Hugging Face (`trust_remote_code=True`).
- Computes 3D electrode positions from the channel names, then runs REVE in batches of 16 (GPU if available).
- Saves the full token-level embeddings, not a pooled vector.

### Analysis
- **`analyze_embeddings.py`** – averages embeddings over channels and time, standardises, projects with PCA to 2D (`results/pca_embeddings.png`), and runs a 5-fold stratified cross-validated logistic regression (balanced accuracy, macro F1).
- **`CNN.py`** – treats the `(channels × time)` embedding map as a 2D image with `dim` features, trains a 2-layer CNN (30 epochs, AdamW) on a stratified 75/25 split. The scaler is fit on training trials only to avoid leakage. Reports accuracy, balanced accuracy, classification report and confusion matrix.

## Scripts

| File | Purpose |
|------|---------|
| `01_inspect_eeg.py` | Prototype on a single recording (sub-P01, ses-1, run-1) |
| `03_check_data.py` | Sanity checks on `segments_labeled.npz` (shape, NaN/Inf, range, segment duration) |
| `prepare_all_data.py` | Preprocess all recordings into labelled segments |
| `embeddings.py` | Single-file embedding extraction (whole dataset in one pass) |
| `embeddings_all_data.py` | Batched embedding extraction, for the full dataset |
| `analyze_embeddings.py` | PCA + linear probe |
| `CNN.py` | CNN classifier on embeddings |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install numpy pandas mne torch transformers scikit-learn matplotlib
```

A CUDA GPU is recommended for embedding extraction but not required.

## Usage

```bash
python prepare_all_data.py        # → data/segments_labeled.npz
python 03_check_data.py           # optional sanity check
python embeddings_all_data.py     # → data/reve_embeddings.npz
python analyze_embeddings.py      # → results/pca_embeddings.png + CV scores
python CNN.py                     # train / evaluate CNN
```

`data/`, `raw_data/`, `hf-cache/` and `results/` outputs are git-ignored (see `.gitignore`).

## Caveats

- Both probes use **trial-level splits**, so trials from the same subject/session can appear in train and test. Scores may be optimistic compared with a leave-one-subject-out evaluation; `segments_metadata.csv` contains the subject IDs needed to do that.
- Averaging over channels and time in `analyze_embeddings.py` discards spatial and temporal structure; the CNN keeps it.
- Segments with a different length from the first are silently dropped.

## Reference

REVE model: [`brain-bzh/reve-base`](https://huggingface.co/brain-bzh/reve-base) on Hugging Face.
