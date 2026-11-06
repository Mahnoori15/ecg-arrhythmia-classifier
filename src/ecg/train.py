"""Train on DS1 patients, validate on held-out DS1 patients, test on unseen DS2 patients.

Usage:
    python -m ecg.train --epochs 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import CLASSES, DS1, DS2, download, load_split
from .evaluate import report
from .model import ECGNet

VAL_PATIENTS = [106, 114, 203, 223]  # held out from DS1 for early stopping


def make_loader(X, F, y, batch=256, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(F), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def class_weights(y, n=len(CLASSES)):
    counts = np.bincount(y, minlength=n).astype(float)
    w = np.where(counts > 0, counts.sum() / (n * np.maximum(counts, 1)), 0.0)
    return torch.tensor(np.sqrt(w), dtype=torch.float32)  # sqrt softens extreme weights


@torch.no_grad()
def predict(model, loader):
    model.eval()
    return np.concatenate([model(xb, fb).argmax(1).numpy() for xb, fb, _ in loader])


def fit(model, train, val, y_train, y_val, epochs=15, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train))
    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, fb, yb in train:
            opt.zero_grad()
            loss = loss_fn(model(xb, fb), yb)
            loss.backward()
            opt.step()
        f1 = report(y_val, predict(model, val))["macro_f1"]
        print(f"epoch {epoch:2d}  loss {loss.item():.3f}  val macro-F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/mitdb")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    data_dir = download(a.data)
    X, F, y, pid = load_split(DS1, data_dir)
    val_mask = np.isin(pid, VAL_PATIENTS)
    Xt, Ft, yt, _ = load_split(DS2, data_dir)
    print(f"train beats {(~val_mask).sum()}  val beats {val_mask.sum()}  test beats {len(yt)}")

    model = fit(
        ECGNet(),
        make_loader(X[~val_mask], F[~val_mask], y[~val_mask], shuffle=True),
        make_loader(X[val_mask], F[val_mask], y[val_mask]),
        y[~val_mask], y[val_mask], epochs=a.epochs,
    )

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "ecgnet.pt")
    res = report(yt, predict(model, make_loader(Xt, Ft, yt)), plot_path=out / "confusion_matrix.png")
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
