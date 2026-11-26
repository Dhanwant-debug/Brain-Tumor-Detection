"""5-fold cross-validation of the custom CNN: a much more reliable accuracy estimate than one
38-image test set.   Run:  python cv_cnn.py      (takes several minutes)"""
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score
import config as cfg
from data import load_arrays
from models import build_custom_cnn

INFO = cfg.MODELS["Custom CNN"]
X, y = load_arrays(INFO["img_size"], INFO["crop"])
X = X.astype("float32")
rows = []
for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=cfg.SEED).split(X, y), 1):
    tf.keras.utils.set_random_seed(cfg.SEED + fold)
    n0, n1 = (y[tr] == 0).sum(), (y[tr] == 1).sum()
    cw = {0: len(tr) / (2 * n0), 1: len(tr) / (2 * n1)}
    model = build_custom_cnn(INFO["img_size"])
    steps = math.ceil(len(tr) / cfg.BATCH)
    lr = tf.keras.optimizers.schedules.CosineDecay(2e-3, cfg.EPOCHS_CNN * steps, alpha=0.01)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X[tr], y[tr].astype("float32"), epochs=cfg.EPOCHS_CNN, batch_size=cfg.BATCH, class_weight=cw, verbose=0)
    pred = (model.predict(X[te], verbose=0).ravel() > 0.5).astype(int)
    r = {"Fold": fold, "Accuracy": accuracy_score(y[te], pred), "Recall (Tumor)": recall_score(y[te], pred),
         "F1 (Tumor)": f1_score(y[te], pred)}
    rows.append(r)
    print(f"Fold {fold}: accuracy {r['Accuracy']:.3f} | recall {r['Recall (Tumor)']:.3f} | F1 {r['F1 (Tumor)']:.3f}")

df = pd.DataFrame(rows).set_index("Fold")
df.loc["Mean"] = df.mean()
df.loc["Std"] = df.iloc[:-1].std()
print("\n", df.round(3).to_string())
df.round(4).to_csv(f"{cfg.RESULTS_DIR}/cnn_cross_validation.csv")
