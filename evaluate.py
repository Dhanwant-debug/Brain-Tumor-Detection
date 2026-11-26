"""Test-set evaluation helpers, used by the training scripts and by compare.py."""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
import config as cfg
import models  # noqa: F401  (registers the custom PerImageStandardize layer for loading)


def predict_test(model_path, test_ds):
    """Load a saved model and return (probabilities, milliseconds per image, parameter count)."""
    model = tf.keras.models.load_model(model_path, compile=False)
    t0 = time.time()
    probs = model.predict(test_ds, verbose=0).ravel()
    ms = (time.time() - t0) / len(probs) * 1000
    return probs, ms, model.count_params()


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    both = len(set(y_true.tolist())) == 2
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Tumor)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Tumor)": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "F1 (Tumor)": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if both else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def report(display_name, key, path, data):
    """Print the classification report and save the confusion matrix for one model."""
    probs, _, _ = predict_test(path, data["test"])
    y_true, y_pred = data["test_labels"], (probs > 0.5).astype(int)
    print(f"\n{display_name} - test results")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=cfg.CLASS_NAMES, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=cfg.CLASS_NAMES).plot(cmap="Blues")
    plt.title(display_name)
    out = f"{cfg.RESULTS_DIR}/{key}_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved model: {path} | confusion matrix: {out}")
