"""Compare all trained models on the SAME held-out test set.
Run after training both models:  python compare.py
Creates results/comparison.csv and comparison plots."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
import os
import config as cfg
from data import get_data
from evaluate import predict_test, compute_metrics
from utils import load_run_info

y_true = None

rows, probs_by_model, infos = [], {}, {}
for name, m in cfg.MODELS.items():
    if not os.path.exists(m["path"]):
        print(f"Skipping {name}: {m['path']} not found (train it first).")
        continue
    data = get_data(m["img_size"], m["crop"])          # each model gets its own preprocessing, same split
    if y_true is None:
        y_true = data["test_labels"]
    assert (y_true == data["test_labels"]).all(), "Models must be evaluated on the same test images"
    probs, ms, n_params = predict_test(m["path"], data["test"])
    info = load_run_info(m["key"]) or {}
    probs_by_model[name], infos[name] = probs, info
    row = {"Model": name, **compute_metrics(y_true, probs)}
    row.update({"Parameters": info.get("params") or n_params, "Epochs run": info.get("epochs_run"),
                "Train time (s)": info.get("train_time_sec"), "Inference (ms/img)": round(ms, 1)})
    rows.append(row)

if not rows:
    raise SystemExit("No trained models found. Run train_mobilenet.py and/or train_cnn.py first.")

df = pd.DataFrame(rows).set_index("Model")
df.round(4).to_csv(f"{cfg.RESULTS_DIR}/comparison.csv")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
print("\n=== MODEL COMPARISON (test set: %d images) ===" % len(y_true))
print(df.round(3).astype(object).T.to_string())

names = list(df.index)
colors = ["#1F3A5F", "#E07B39", "#4C9F70"][:len(names)]

# 1) grouped bar chart of main metrics
metrics = ["Accuracy", "Precision (Tumor)", "Recall (Tumor)", "Specificity", "F1 (Tumor)", "ROC-AUC"]
x = np.arange(len(metrics)); w = 0.8 / len(names)
plt.figure(figsize=(10, 4.5))
for i, n in enumerate(names):
    plt.bar(x + i * w, [df.loc[n, mt] for mt in metrics], w, label=n, color=colors[i])
plt.xticks(x + w * (len(names) - 1) / 2, metrics, rotation=20, ha="right")
plt.ylim(0, 1.05); plt.ylabel("Score"); plt.title("Model comparison on the test set"); plt.legend()
plt.savefig(f"{cfg.RESULTS_DIR}/comparison_metrics.png", dpi=150, bbox_inches="tight"); plt.close()

# 2) confusion matrices side by side
fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4), squeeze=False)
for ax, n in zip(axes[0], names):
    cm = confusion_matrix(y_true, (probs_by_model[n] > 0.5).astype(int), labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=cfg.CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(n)
plt.savefig(f"{cfg.RESULTS_DIR}/comparison_confusion.png", dpi=150, bbox_inches="tight"); plt.close()

# 3) ROC curves
plt.figure(figsize=(5, 4.5))
for n, c in zip(names, colors):
    if len(set(y_true.tolist())) == 2:
        fpr, tpr, _ = roc_curve(y_true, probs_by_model[n])
        plt.plot(fpr, tpr, color=c, label=f"{n} (AUC {df.loc[n, 'ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate (recall)")
plt.title("ROC curves"); plt.legend(loc="lower right")
plt.savefig(f"{cfg.RESULTS_DIR}/comparison_roc.png", dpi=150, bbox_inches="tight"); plt.close()

# 4) training curves (validation accuracy and loss)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for n, c in zip(names, colors):
    h = infos[n].get("history")
    if h:
        axes[0].plot(h["val_accuracy"], color=c, label=n); axes[1].plot(h["val_loss"], color=c, label=n)
axes[0].set_title("Validation accuracy"); axes[1].set_title("Validation loss")
for a in axes: a.set_xlabel("Epoch"); a.legend()
plt.savefig(f"{cfg.RESULTS_DIR}/comparison_training.png", dpi=150, bbox_inches="tight"); plt.close()

print(f"\nSaved table and 4 plots in '{cfg.RESULTS_DIR}/'")
