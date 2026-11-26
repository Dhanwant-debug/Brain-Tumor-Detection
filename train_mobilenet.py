"""Train model 1: MobileNetV2 transfer learning.   Run:  python train_mobilenet.py"""
import time
import tensorflow as tf
import config as cfg
from data import get_data
from models import build_mobilenet
from utils import save_run_info
from evaluate import report

NAME = "MobileNetV2"
INFO = cfg.MODELS[NAME]
tf.keras.utils.set_random_seed(cfg.SEED)

data = get_data(INFO["img_size"], INFO["crop"])
model, base = build_mobilenet(INFO["img_size"])
t0 = time.time()

# ---- Phase 1: train the new head, base frozen ----
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
h1 = model.fit(
    data["train"], validation_data=data["val"], epochs=cfg.EPOCHS_HEAD, class_weight=data["class_weight"],
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
               tf.keras.callbacks.ModelCheckpoint(INFO["path"], monitor="val_loss", save_best_only=True)])
best_val_loss = min(h1.history["val_loss"])

# ---- Phase 2: fine-tune the top 30 layers (kept only if validation loss improves) ----
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
h2 = model.fit(
    data["train"], validation_data=data["val"], epochs=cfg.EPOCHS_FINE, class_weight=data["class_weight"],
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
               tf.keras.callbacks.ModelCheckpoint(INFO["path"], monitor="val_loss", save_best_only=True,
                                                  initial_value_threshold=best_val_loss)])

save_run_info(INFO["key"], [h1.history, h2.history], time.time() - t0, model)
report(NAME, INFO["key"], INFO["path"], data)
