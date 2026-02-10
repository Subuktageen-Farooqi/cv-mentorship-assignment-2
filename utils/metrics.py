from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred)

    # per-class precision/recall/f1/support
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    # per-class accuracy = diagonal / row sum
    row_sum = cm.sum(axis=1).clip(min=1)
    per_class_acc = np.diag(cm) / row_sum

    df_class = pd.DataFrame({
        "class": class_names,
        "accuracy": per_class_acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    }).set_index("class")

    overall_acc = accuracy_score(y_true, y_pred)

    def agg(avg: str):
        ap, ar, af1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        return ap, ar, af1

    macro = agg("macro")
    micro = agg("micro")
    weighted = agg("weighted")

    df_global = pd.DataFrame([
        {"avg": "accuracy", "precision": np.nan, "recall": np.nan, "f1": np.nan, "value": overall_acc},
        {"avg": "macro", "precision": macro[0], "recall": macro[1], "f1": macro[2], "value": np.nan},
        {"avg": "micro", "precision": micro[0], "recall": micro[1], "f1": micro[2], "value": np.nan},
        {"avg": "weighted", "precision": weighted[0], "recall": weighted[1], "f1": weighted[2], "value": np.nan},
    ]).set_index("avg")

    return df_class, df_global, cm

def save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
