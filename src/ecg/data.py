"""MIT-BIH data loading, filtering, beat segmentation and the inter-patient split.

Classes follow the AAMI EC57 standard:
    N = normal / bundle branch block, S = supraventricular ectopic,
    V = ventricular ectopic, F = fusion, Q = paced / unclassifiable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

FS = 360  # MIT-BIH sampling rate (Hz)
BEFORE, AFTER = 90, 166  # samples around the R peak -> 256-sample window (~0.71 s)
CLASSES = ["N", "S", "V", "F", "Q"]

AAMI = {
    **dict.fromkeys(["N", "L", "R", "e", "j"], "N"),
    **dict.fromkeys(["A", "a", "J", "S"], "S"),
    **dict.fromkeys(["V", "E"], "V"),
    "F": "F",
    **dict.fromkeys(["/", "f", "Q"], "Q"),
}

# de Chazal et al. (2004) inter-patient split. Paced records 102, 104, 107, 217 are excluded.
# Training and test beats come from DIFFERENT patients, which is how a real deployment works.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


def download(data_dir: str | Path = "data/mitdb") -> Path:
    """Download MIT-BIH Arrhythmia Database from PhysioNet (~100 MB)."""
    import wfdb

    data_dir = Path(data_dir)
    if not (data_dir / "100.dat").exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database("mitdb", dl_dir=str(data_dir))
    return data_dir


def bandpass(signal: np.ndarray, low: float = 0.5, high: float = 40.0, fs: int = FS) -> np.ndarray:
    """Remove baseline wander (<0.5 Hz) and high-frequency noise (>40 Hz)."""
    b, a = butter(3, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, signal)


def rr_features(r_peaks: np.ndarray) -> np.ndarray:
    """Per-beat RR features, normalised by the patient's mean RR so they transfer across patients.

    Returns (n_beats, 4): pre-RR, post-RR, local mean RR (last 10 beats), pre/post ratio.
    The first and last beats get NaN and are dropped by the caller.
    """
    rr = np.diff(r_peaks).astype(float)
    mean_rr = rr.mean()
    n = len(r_peaks)
    feats = np.full((n, 4), np.nan)
    for i in range(1, n - 1):
        pre, post = rr[i - 1], rr[i]
        local = rr[max(0, i - 10):i].mean()
        feats[i] = [pre / mean_rr, post / mean_rr, local / mean_rr, pre / post]
    return feats


def segment_beats(signal, r_peaks, symbols):
    """Cut fixed windows around each labelled beat. Returns (beats, rr_feats, labels)."""
    signal = bandpass(signal)
    feats = rr_features(r_peaks)
    X, F, y = [], [], []
    for i, (r, s) in enumerate(zip(r_peaks, symbols)):
        if s not in AAMI or np.isnan(feats[i]).any():
            continue
        if r - BEFORE < 0 or r + AFTER > len(signal):
            continue
        w = signal[r - BEFORE:r + AFTER]
        w = (w - w.mean()) / (w.std() + 1e-8)  # per-beat z-score
        X.append(w)
        F.append(feats[i])
        y.append(CLASSES.index(AAMI[s]))
    return np.array(X, np.float32), np.array(F, np.float32), np.array(y, np.int64)


def load_record(record: int, data_dir: str | Path):
    import wfdb

    path = str(Path(data_dir) / str(record))
    rec = wfdb.rdrecord(path, channels=[0])  # lead MLII
    ann = wfdb.rdann(path, "atr")
    beat_mask = np.isin(ann.symbol, list(AAMI))
    r_peaks = ann.sample[beat_mask]
    symbols = np.array(ann.symbol)[beat_mask]
    return segment_beats(rec.p_signal[:, 0], r_peaks, symbols)


def load_split(records, data_dir):
    """Load many records. Also returns a patient id per beat (for patient-wise validation)."""
    parts = [load_record(r, data_dir) for r in records]
    X = np.concatenate([p[0] for p in parts])
    F = np.concatenate([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts])
    pid = np.concatenate([np.full(len(p[2]), r) for p, r in zip(parts, records)])
    return X, F, y, pid
