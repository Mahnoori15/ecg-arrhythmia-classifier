"""Per-class metrics and confusion matrix. Accuracy alone is misleading here:
about 90% of beats are normal, so a model that always says "N" scores ~90%."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .data import CLASSES


def report(y_true, y_pred, plot_path=None) -> dict:
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    present = [i for i in labels if s[i] > 0]
    res = {
        "accuracy": round(float((np.asarray(y_true) == np.asarray(y_pred)).mean()), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)), 4),
        "per_class": {
            CLASSES[i]: {"precision": round(float(p[i]), 4), "recall (sensitivity)": round(float(r[i]), 4),
                         "f1": round(float(f[i]), 4), "support": int(s[i])}
            for i in labels
        },
        "confusion_matrix": cm.tolist(),
    }
    if plot_path:
        _plot(cm, plot_path)
    return res


def _plot(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Inter-patient test (DS2), row-normalised")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
