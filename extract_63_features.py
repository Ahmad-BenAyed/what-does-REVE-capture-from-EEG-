
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.signal import hilbert
from scipy.fft import rfft, irfft, rfftfreq
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

FILE = "segments_labeled.npz"
FS = 200
EPS = 1e-12

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

# ============================================================
# FFT RECTANGULAR FILTER
# ============================================================

def fft_bandpass(x, fs, low, high):
    n = x.shape[-1]
    freqs = rfftfreq(n, 1 / fs)
    spectrum = rfft(x)
    mask = (freqs >= low) & (freqs < high)
    return irfft(spectrum * mask, n=n)

def band_energy(x, fs, low, high):
    spectrum = np.abs(rfft(x)) ** 2
    freqs = rfftfreq(x.size, 1 / fs)
    mask = (freqs >= low) & (freqs < high)
    return spectrum[mask].sum()

# ============================================================
# FAMILY T: TIME DOMAIN (10)
# ============================================================

def family_T(x):
    dx = np.diff(x) * FS
    ddx = np.diff(dx) * FS

    vx = np.var(x)
    vdx = np.var(dx)
    vddx = np.var(ddx)

    mobility = np.sqrt(vdx / (vx + EPS))
    mobility_dx = np.sqrt(vddx / (vdx + EPS))

    return [
        vx,
        mobility,
        mobility_dx / (mobility + EPS),
        np.sqrt(vx),
        np.sqrt(np.mean(x ** 2)),
        kurtosis(x, fisher=False, bias=False),
        np.mean(x[:-1] * x[1:] < 0),
        np.mean(np.abs(np.diff(x))),
        np.sqrt(vdx),
        np.ptp(x)
    ]

# ============================================================
# FAMILY F: FREQUENCY DOMAIN (16)
# ============================================================

def family_F(x):
    energies = {
        name: band_energy(x, FS, *limits)
        for name, limits in BANDS.items()
    }

    names = list(BANDS.keys())
    powers = np.array([energies[n] for n in names])
    total = powers.sum() + EPS

    log_power = np.log(powers + EPS)
    relative_power = powers / total

    ratios = [
        np.log((energies["theta"] + EPS) /
               (energies["beta"] + EPS)),
        np.log((energies["delta"] + EPS) /
               (energies["alpha"] + EPS)),
        np.log((energies["theta"] + EPS) /
               (energies["alpha"] + EPS))
    ]

    spectrum = np.abs(rfft(x)) ** 2
    freqs = rfftfreq(len(x), 1 / FS)

    mask = (
        (freqs >= 0.5) & (freqs < 45)
    )
    p = spectrum[mask]
    f = freqs[mask]

    p = p / (p.sum() + EPS)

    entropy = -np.sum(p * np.log(p + EPS))
    entropy /= np.log(max(len(p), 2))

    centroid = np.sum(f * p)

    cumulative = np.cumsum(p)
    edge = f[min(np.searchsorted(cumulative, 0.95), len(f)-1)]

    return list(log_power) + list(relative_power) + ratios + [
        entropy, centroid, edge
    ]

# ============================================================
# FAMILY TF: TIME-FREQUENCY (11)
# ============================================================

def haar_details(x):
    current = x.copy()
    details = []

    for _ in range(5):
        n = len(current)
        if n % 2:
            current = current[:-1]

        even = current[::2]
        odd = current[1::2]

        detail = (even - odd) / np.sqrt(2)
        approximation = (even + odd) / np.sqrt(2)

        details.append(detail)
        current = approximation

    return details, current

def family_TF(x):
    details, approximation = haar_details(x)

    energies = [
        np.mean(d ** 2) for d in details
    ] + [np.mean(approximation ** 2)]

    energies = np.array(energies)
    probabilities = energies / (energies.sum() + EPS)

    entropy = -np.sum(
        probabilities * np.log(probabilities + EPS)
    ) / np.log(6)

    detail_variances = [
        np.var(d) for d in details
    ]

    envelope_cv = []

    for low, high in BANDS.values():
        filtered = fft_bandpass(x, FS, low, high)
        envelope = np.abs(hilbert(filtered))

        cv = np.std(envelope) / (np.mean(envelope) + EPS)
        envelope_cv.append(cv)

    return [entropy] + detail_variances + envelope_cv

# ============================================================
# FAMILY C: COMPLEXITY (7)
# ============================================================

def family_C(x):
    # C001: normalized permutation entropy, m=3
    patterns = np.lib.stride_tricks.sliding_window_view(x, 3)
    patterns = np.argsort(patterns, axis=1, kind="stable")

    _, counts = np.unique(
        patterns, axis=0, return_counts=True
    )
    p = counts / counts.sum()
    pe = -np.sum(p * np.log(p + EPS)) / np.log(6)

    # C002: derivative-irregularity proxy
    dx = np.diff(x)
    ddx = np.diff(dx)
    r = 0.2 * np.std(x)

    c002 = (
        np.log1p(np.mean(np.abs(ddx)) / (r + EPS))
        - np.log1p(np.mean(np.abs(dx)) / (r + EPS))
    )

    # C003: LZ-style binarized irregularity proxy
    b = (x > np.median(x)).astype(int)
    pairs = b[:-1] * 2 + b[1:]

    _, counts = np.unique(pairs, return_counts=True)
    p = counts / counts.sum()

    h2 = -np.sum(p * np.log(p + EPS)) / np.log(4)
    eta = np.mean(b[1:] != b[:-1])
    c003 = 0.5 * eta + 0.5 * h2

    # C004: Higuchi-style lag-difference slope proxy
    kmax = min(8, len(x) // 4)
    ks = np.arange(1, kmax + 1)

    lag_means = np.array([
        np.mean(np.abs(x[k:] - x[:-k]))
        for k in ks
    ])

    valid = lag_means > EPS
    if valid.sum() >= 2:
        c004 = np.polyfit(
            np.log(1 / ks[valid]),
            np.log(lag_means[valid]), 1
        )[0]
    else:
        c004 = 0.0

    c004 = np.clip(c004, 0, 3)

    # C005: DFA-style exponent proxy
    y = np.cumsum(x - np.mean(x))
    scales = [16, 32, 64, 128, 256, 512]
    scales = [s for s in scales if 2 * s <= len(x)]

    fluctuations = []

    for s in scales:
        n_windows = len(y) // s
        residuals = []

        for j in range(n_windows):
            segment = y[j*s:(j+1)*s]
            t = np.arange(s)
            trend = np.polyval(
                np.polyfit(t, segment, 1), t
            )
            residuals.extend(segment - trend)

        fluctuations.append(np.sqrt(np.mean(np.array(residuals) ** 2)))

    if len(fluctuations) >= 2 and np.all(np.array(fluctuations) > EPS):
        c005 = np.polyfit(
            np.log(scales), np.log(fluctuations), 1
        )[0]
    else:
        c005 = 0.0

    c005 = np.clip(c005, -1, 2)

    # C006 and C007: FFT-based autocorrelation
    centered = x - np.mean(x)
    n = len(centered)

    spectrum = np.fft.rfft(centered, n=2*n)
    acf = np.fft.irfft(
        spectrum * np.conj(spectrum)
    )[:n]

    acf /= acf[0] + EPS

    tau_max = min(n // 2, int(FS * 2))
    acf = acf[:tau_max + 1]

    decay = np.where(acf[1:] <= 1 / np.e)[0]
    zero = np.where(acf[1:] <= 0)[0]

    tau_e = decay[0] + 1 if len(decay) else tau_max
    tau_0 = zero[0] + 1 if len(zero) else tau_max

    c006 = tau_e / tau_max
    c007 = tau_0 / tau_max

    return [
        pe, c002, c003, c004, c005, c006, c007
    ]

# ============================================================
# FAMILY X: CROSS-FREQUENCY COUPLING (5)
# ============================================================

def modulation_index(phase, amplitude, bins=18):
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    mean_amplitudes = []

    for j in range(bins):
        mask = (
            (phase >= edges[j]) &
            (phase < edges[j+1])
        )

        mean_amplitudes.append(
            np.mean(amplitude[mask]) if np.any(mask) else 0.0
        )

    p = np.array(mean_amplitudes)
    p /= p.sum() + EPS

    return np.sum(
        p * np.log((p + EPS) * bins)
    ) / np.log(bins)

def family_X(x):
    envelopes = {}
    phases = {}

    for name, limits in BANDS.items():
        filtered = fft_bandpass(x, FS, *limits)
        analytic = hilbert(filtered)

        envelopes[name] = np.abs(analytic)
        phases[name] = np.angle(analytic)

    x001 = modulation_index(
        phases["delta"], envelopes["beta"]
    )
    x002 = modulation_index(
        phases["theta"], envelopes["gamma"]
    )
    x003 = modulation_index(
        phases["alpha"], envelopes["gamma"]
    )

    x004 = np.corrcoef(
        envelopes["theta"], envelopes["gamma"]
    )[0, 1]

    x005 = np.corrcoef(
        envelopes["delta"], envelopes["gamma"]
    )[0, 1]

    return [
        x001, x002, x003,
        np.clip(x004, -1, 1),
        np.clip(x005, -1, 1)
    ]

# ============================================================
# FAMILY R: CROSS-CHANNEL (14)
# ============================================================

def family_R(epoch):
    n_channels, n_time = epoch.shape

    # Population Pearson correlation matrix
    corr = np.corrcoef(epoch)
    pairs = np.triu_indices(n_channels, k=1)
    pair_corr = corr[pairs]

    r001 = np.mean(np.abs(pair_corr))
    r002 = np.std(np.abs(pair_corr), ddof=0)

    # Eigenvalues of signed correlation matrix
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues, EPS, None)
    eigenvalues /= eigenvalues.sum()

    r003 = -np.sum(
        eigenvalues * np.log(eigenvalues)
    ) / np.log(n_channels)

    r004 = (
        eigenvalues.sum() ** 2 /
        (np.sum(eigenvalues ** 2) + EPS)
    )

    coherence = []
    pli = []

    for name, (low, high) in BANDS.items():
        filtered = np.array([
            fft_bandpass(ch, FS, low, high)
            for ch in epoch
        ])

        analytic = hilbert(filtered, axis=-1)
        phase = np.angle(analytic)

        # FFT-bin magnitude coherence proxy
        spectra = rfft(filtered, axis=-1)
        freqs = rfftfreq(n_time, 1 / FS)
        mask = (freqs >= low) & (freqs < high)

        band_spectra = spectra[:, mask]

        pair_coherence = []

        for i, j in combinations(range(n_channels), 2):
            sii = np.mean(np.abs(band_spectra[i]) ** 2)
            sjj = np.mean(np.abs(band_spectra[j]) ** 2)

            sij = np.mean(
                band_spectra[i] *
                np.conj(band_spectra[j])
            )

            value = np.abs(sij) / (
                np.sqrt(sii * sjj) + EPS
            )
            pair_coherence.append(np.clip(value, 0, 1))

        coherence.append(np.mean(pair_coherence))

        pair_pli = []

        for i, j in combinations(range(n_channels), 2):
            delta = phase[i] - phase[j]
            value = np.abs(np.mean(np.sign(np.sin(delta))))
            pair_pli.append(value)

        pli.append(np.mean(pair_pli))

    return [r001, r002, r003, r004] + coherence + pli

# ============================================================
# FEATURE REGISTRY
# ============================================================

T_NAMES = [
    f"T{i:03d}" for i in range(1, 11)
]

F_NAMES = [
    f"F{i:03d}" for i in range(1, 17)
]

TF_NAMES = [
    f"TF{i:03d}" for i in range(1, 12)
]

C_NAMES = [
    f"C{i:03d}" for i in range(1, 8)
]

X_NAMES = [
    f"X{i:03d}" for i in range(1, 6)
]

R_NAMES = [
    f"R{i:03d}" for i in range(1, 15)
]

CHANNEL_NAMES = (
    T_NAMES + F_NAMES + TF_NAMES + C_NAMES + X_NAMES
)

GLOBAL_NAMES = R_NAMES

assert len(CHANNEL_NAMES) == 49
assert len(GLOBAL_NAMES) == 14

# ============================================================
# LOAD DATA
# ============================================================

data = np.load(FILE, allow_pickle=True)

print("NPZ keys:", data.files)

signal_candidates = [
    k for k in data.files
    if k.lower() in ("segments", "x", "eeg", "signals", "data")
    and data[k].ndim in (2, 3)
    and data[k].dtype.kind in "iuf"
]

if len(signal_candidates) != 1:
    raise ValueError(
        f"Signal key ambiguous: {signal_candidates}"
    )

X = np.asarray(data[signal_candidates[0]], dtype=np.float64)

if X.ndim == 2:
    X = X[:, None, :]

if X.ndim != 3:
    raise ValueError(f"Unexpected EEG shape: {X.shape}")

N, C, T = X.shape

print("\nSegments:", N)
print("Channels:", C)
print("Samples:", T)
print("Duration:", T / FS, "seconds")

label_keys = [
    k for k in data.files
    if k.lower() in ("y", "labels", "label", "targets")
]

labels = (
    data[label_keys[0]]
    if len(label_keys) == 1 else np.array([])
)

# ============================================================
# EXTRACT ALL 63 FEATURES
# ============================================================

per_channel = np.zeros((N, C, 49), dtype=np.float32)
global_features = np.zeros((N, 14), dtype=np.float32)

for i in range(N):
    if (i + 1) % 100 == 0:
        print(f"Processing {i+1}/{N}")

    epoch = X[i]

    for ch in range(C):
        signal = epoch[ch]

        values = (
            family_T(signal)
            + family_F(signal)
            + family_TF(signal)
            + family_C(signal)
            + family_X(signal)
        )

        per_channel[i, ch] = np.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        )

    global_values = family_R(epoch)

    global_features[i] = np.nan_to_num(
        global_values, nan=0.0, posinf=0.0, neginf=0.0
    )

# ============================================================
# SAVE NPZ
# ============================================================

np.savez_compressed(
    "features_63.npz",
    per_channel=per_channel,
    global_features=global_features,
    channel_feature_names=np.array(CHANNEL_NAMES),
    global_feature_names=np.array(GLOBAL_NAMES),
    labels=labels,
    sampling_frequency=FS
)

# ============================================================
# SAVE CSV: ONE ROW PER SEGMENT AND CHANNEL
# ============================================================

rows = []

for i in range(N):
    for ch in range(C):
        row = {
            "segment": i,
            "channel": ch
        }

        if len(labels) == N:
            row["label"] = labels[i]

        row.update(
            dict(zip(CHANNEL_NAMES, per_channel[i, ch]))
        )

        rows.append(row)

pd.DataFrame(rows).to_csv(
    "features_49_per_channel.csv",
    index=False
)

# One row per segment for global R features
global_df = pd.DataFrame(
    global_features,
    columns=GLOBAL_NAMES
)
global_df.insert(0, "segment", np.arange(N))

global_df.to_csv(
    "features_14_global.csv",
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n========== EXTRACTION COMPLETED ==========")
print("Per-channel features:", per_channel.shape)
print("Global features:", global_features.shape)
print("Total base features:", 49 + 14)

print("\nSaved:")
print("1. features_63.npz")
print("2. features_49_per_channel.csv")
print("3. features_14_global.csv")

