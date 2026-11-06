"""Tests on synthetic ECG so CI runs without downloading MIT-BIH."""
import numpy as np
import torch

from ecg.data import AAMI, BEFORE, AFTER, CLASSES, DS1, DS2, rr_features, segment_beats
from ecg.evaluate import report
from ecg.model import ECGNet
from ecg.train import fit, make_loader

rng = np.random.default_rng(0)


def synthetic_ecg(n_beats=300, fs=360):
    """Normal beats are narrow and regular; 'A' beats come early; 'V' beats are wide and tall."""
    t, peaks, syms = 400, [], []
    for _ in range(n_beats):
        kind = rng.choice(["N", "A", "V"], p=[0.6, 0.2, 0.2])
        t += int(fs * (0.55 if kind == "A" else 0.85))
        peaks.append(t)
        syms.append(kind)
    sig = rng.normal(0, 0.02, t + 400)
    x = np.arange(-60, 60)
    for p, s in zip(peaks, syms):
        width, amp = (12, 1.8) if s == "V" else (3, 1.0)
        sig[p - 60:p + 60] += amp * np.exp(-(x / width) ** 2)
    return sig, np.array(peaks), np.array(syms)


def test_split_is_inter_patient():
    assert not set(DS1) & set(DS2)
    assert len(DS1) == len(DS2) == 22


def test_aami_mapping_covers_all_classes():
    assert set(AAMI.values()) == set(CLASSES)


def test_rr_features_flag_premature_beat():
    peaks = np.array([0, 300, 600, 750, 1050, 1350])  # beat 3 is early
    f = rr_features(peaks)
    assert np.isnan(f[0]).all() and np.isnan(f[-1]).all()
    assert f[3, 0] < f[2, 0]  # shorter pre-RR for the premature beat


def test_segmentation_shapes():
    sig, peaks, syms = synthetic_ecg(50)
    X, F, y = segment_beats(sig, peaks, syms)
    assert X.shape[1] == BEFORE + AFTER and F.shape[1] == 4
    assert len(X) == len(F) == len(y) == 48  # first and last beats dropped


def test_model_output_shape():
    out = ECGNet()(torch.randn(8, BEFORE + AFTER), torch.randn(8, 4))
    assert out.shape == (8, len(CLASSES))


def test_model_learns_synthetic_beats():
    torch.manual_seed(0)
    X, F, y = segment_beats(*synthetic_ecg(400))
    Xv, Fv, yv = segment_beats(*synthetic_ecg(150))
    model = fit(ECGNet(), make_loader(X, F, y, 32, True), make_loader(Xv, Fv, yv), y, yv, epochs=15)
    from ecg.train import predict
    assert report(yv, predict(model, make_loader(Xv, Fv, yv)))["macro_f1"] > 0.9


def test_report_perfect_predictions():
    y = np.array([0, 0, 1, 2, 2])
    r = report(y, y)
    assert r["accuracy"] == 1.0 and r["macro_f1"] == 1.0
