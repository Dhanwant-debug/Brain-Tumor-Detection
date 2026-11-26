"""Data loading: one stratified 70/15/15 split shared by BOTH models (fair comparison).
Images are loaded once into memory (the dataset is small), which also makes training faster."""
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import config as cfg
from preprocess import prepare

AUTOTUNE = tf.data.AUTOTUNE


def load_arrays(img_size, crop):
    """Returns X (n, size, size, 3) uint8 and y (n,) with 0 = no tumor, 1 = tumor."""
    paths, labels = [], []
    for label, cls in enumerate(["no", "yes"]):
        for f in sorted(glob.glob(os.path.join(cfg.DATA_DIR, cls, "*"))):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(f)
                labels.append(label)
    if not paths:
        raise SystemExit(f"No images found. Expected folders '{cfg.DATA_DIR}/yes' and '{cfg.DATA_DIR}/no'.")
    X = np.stack([prepare(Image.open(f), img_size, crop) for f in paths])
    return X, np.array(labels)


def _make_ds(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y.astype("float32").reshape(-1, 1)))
    if shuffle:
        ds = ds.shuffle(len(X), seed=cfg.SEED)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=AUTOTUNE)
    return ds.batch(cfg.BATCH).prefetch(AUTOTUNE)


def split_indices(y):
    """Stratified 70/15/15 split. Same indices for every model because the file order and seed are fixed."""
    idx = np.arange(len(y))
    train, temp = train_test_split(idx, test_size=0.30, stratify=y, random_state=cfg.SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=y[temp], random_state=cfg.SEED)
    return train, val, test


def get_data(img_size, crop=False):
    """Returns a dict with train/val/test datasets, test labels, class weights and train size."""
    X, y = load_arrays(img_size, crop)
    tr, va, te = split_indices(y)
    print(f"Found {len(y)} images | no tumor: {(y == 0).sum()} | tumor: {(y == 1).sum()} "
          f"| input {img_size}x{img_size}, brain crop: {crop}")
    print(f"Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")
    n0, n1 = (y[tr] == 0).sum(), (y[tr] == 1).sum()
    return {
        "train": _make_ds(X[tr], y[tr], shuffle=True),
        "val": _make_ds(X[va], y[va]),
        "test": _make_ds(X[te], y[te]),
        "test_labels": y[te],
        "class_weight": {0: len(tr) / (2 * n0), 1: len(tr) / (2 * n1)},
        "n_train": len(tr),
    }
