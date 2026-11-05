# What does REVE capture from EEG?

Exploration of what the representations of **REVE** (`brain-bzh/reve-base`), a pretrained EEG foundation model, encode. The project:

1. cuts a 3-class mental-task EEG dataset into labelled trials,
2. extracts REVE token-level embeddings for every trial,
3. computes **63 hand-crafted EEG features** (time, frequency, time-frequency, complexity, cross-frequency coupling, cross-channel),
4. fits **ridge regressions from the (pooled) REVE embedding to each feature**, to measure which classical EEG properties are linearly decodable from REVE.

## Task and data

EEG recorded during three mental tasks (BIDS format, dataset id `nm000290`):

| Label | Class | Description |
|-------|-------|-------------|
| 0 | `mi` | Motor imagery |
| 1 | `sub` | Mental subtraction |
| 2 | `word` | Word generation |

- 41 EEG channels (order is hard-coded in `02_embeddings_all_data.py`)
- Trials are cut using the `onset` / `duration` columns of each `*_events.tsv`

## Pipeline

```
raw EDF + events.tsv
      │  01_prepare_all_data.py
      ▼
data/segments_labeled.npz        (X: trials × channels × time, y)
data/segments_metadata.csv
      │
      ├── 02_embeddings_all_data.py   (REVE)
      │        ▼
      │   data/reve_embeddings.npz    (embeddings: trials × channels × patches × dim, y)
      │
      └── 03_extract_63_features.py
               ▼
          features_63.npz, features_49_per_channel.csv, features_14_global.csv
               │
               ▼   (together with reve_embeddings.npz)
          04_regression_63.py
               ▼
          reve_63_ridge_results.csv, reve_63_channel_results.csv, reve_63_family_summary.csv
```

### 01 – Preprocessing (`01_prepare_all_data.py`)
1. Read each EDF file with MNE and resample to **200 Hz** (REVE's expected rate).
2. Band-pass filter **0.5–99.5 Hz**.
3. Z-score each channel per file, then clip to **±15 σ**.
4. Extract one segment per `mi` / `sub` / `word` event; skip invalid or out-of-range events and segments whose shape differs from the first one (needed to stack into one array).

### 02 – Embedding extraction (`02_embeddings_all_data.py`)
- Loads `brain-bzh/reve-base` and the electrode position bank `brain-bzh/reve-positions` from Hugging Face (`trust_remote_code=True`).
- Computes 3D electrode positions from the channel names, then runs REVE in batches of 16 (GPU if available).
- Saves the full token-level embeddings, not a pooled vector.

### 03 – Feature extraction (`03_extract_63_features.py`)
Computes 63 features per trial, in 6 families (sampling rate 200 Hz, bands: delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz):

| Family | Scope | # | Content |
|--------|-------|---|---------|
| **T** – time domain | per channel | 10 | variance, Hjorth mobility / complexity, std, RMS, kurtosis, zero-crossing rate, mean abs. diff, derivative std, peak-to-peak |
| **F** – frequency domain | per channel | 16 | log and relative band power (5+5), θ/β, δ/α, θ/α log-ratios, spectral entropy, centroid, 95 % edge frequency |
| **TF** – time-frequency | per channel | 11 | Haar wavelet entropy, per-level detail variances (5), Hilbert envelope coefficient of variation per band (5) |
| **C** – complexity | per channel | 7 | permutation entropy, derivative-irregularity, LZ-style, Higuchi-style slope, DFA-style exponent, autocorrelation decay times (2) |
| **X** – cross-frequency coupling | per channel | 5 | modulation index (δ→β, θ→γ, α→γ), envelope correlation (θ–γ, δ–γ) |
| **R** – cross-channel | per trial | 14 | mean / std of absolute channel correlation, eigenspectrum entropy and participation ratio, coherence (5 bands), phase-lag index (5 bands) |

That gives 49 per-channel features (`per_channel`, shape trials × 41 × 49) and 14 global features (`global_features`, shape trials × 14). Several complexity features are explicit *proxies* (LZ, Higuchi, DFA), not reference implementations.

### 04 – Ridge probing (`04_regression_63.py`)
- Pools REVE embeddings over channels and patches: `(N, 41, patches, 512) → (N, 512)`, then standardises (scaler fit on train only).
- Random **70 / 15 / 15** train / validation / test split (seed 42).
- For each of the 63 features, fits a Ridge regression from the embedding to the (standardised) feature — for per-channel features the 41 channels are predicted jointly as a multi-output target.
- Selects `alpha` ∈ {0.1, 1, 10, 100} on validation R², refits on train+validation, and reports **test R²** (averaged over channels for per-channel features).
- Outputs: per-feature results, per-channel results, and a per-family summary (mean / std / min / max test R²).

## Scripts

| File | Purpose |
|------|---------|
| `01_prepare_all_data.py` | Preprocess all recordings into labelled segments |
| `02_embeddings_all_data.py` | Batched REVE embedding extraction |
| `03_extract_63_features.py` | Compute the 63 hand-crafted EEG features |
| `04_regression_63.py` | Ridge regression from REVE embeddings to the 63 features |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install numpy pandas scipy mne torch transformers scikit-learn
```

A CUDA GPU is recommended for embedding extraction but not required.

## Usage

```bash
python 01_prepare_all_data.py       # → data/segments_labeled.npz
python 02_embeddings_all_data.py    # → data/reve_embeddings.npz
python 03_extract_63_features.py    # → features_63.npz + 2 CSV files
python 04_regression_63.py          # → reve_63_*.csv
```

**Paths:** `01` and `02` read/write under `data/`, whereas `03` and `04` currently expect `segments_labeled.npz`, `reve_embeddings.npz` and `features_63.npz` in the **working directory**. Either copy/move the two `.npz` files from `data/` to the project root or edit the `FILE` / `EMB_FILE` constants at the top of those scripts.

`data/`, `raw_data/` and `hf-cache/` are git-ignored (see `.gitignore`).

## Caveats

- The train/val/test split in `04` is random at trial level, so trials from the same subject/session can appear in different splits. R² may be optimistic compared with a leave-one-subject-out evaluation; `segments_metadata.csv` contains the subject IDs needed to do that.
- Pooling over channels and time discards spatial and temporal structure of the embeddings; per-channel features are therefore predicted from a single global vector.
- In `04`, the family label is derived with `feature_name.split("_")[0]`, but feature names look like `T001`, `TF003`, … (no underscore), so every per-channel feature ends up as its own "family" in the family summary. Use `re.match(r"[A-Z]+", name)` to group them properly.
- Segments with a different shape from the first are silently dropped in `01`.
- Features are computed on z-scored, clipped signals, so absolute-amplitude features (variance, std, RMS, peak-to-peak) reflect normalised, not physical, amplitude.

## Reference

REVE model: [`brain-bzh/reve-base`](https://huggingface.co/brain-bzh/reve-base) on Hugging Face.
