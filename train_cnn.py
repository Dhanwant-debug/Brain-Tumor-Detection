"""Train model 2: custom CNN from scratch.   Run:  python train_cnn.py

Recipe (what made the from-scratch CNN work on only ~177 training images):
  * crop to the brain + small 64x64 input
  * per-image standardization (inside the model)
  * fixed-length training with cosine learning-rate decay; the FINAL weights are kept
    (restoring the 'best validation loss' epoch kept selecting an early, collapsed model)
  * BatchNorm momentum 0.9, L2 1e-4, dropout 0.3, light augmentation, class weights"""
import math
import time
import tensorflow as tf
import config as cfg
from data import get_data
from models import build_custom_cnn
from utils import save_run_info
from evaluate import report

NAME = "Custom CNN"
INFO = cfg.MODELS[NAME]
tf.keras.utils.set_random_seed(cfg.SEED)

data = get_data(INFO["img_size"], INFO["crop"])
model = build_custom_cnn(INFO["img_size"])
model.summary()
t0 = time.time()

steps = math.ceil(data["n_train"] / cfg.BATCH)
lr = tf.keras.optimizers.schedules.CosineDecay(2e-3, cfg.EPOCHS_CNN * steps, alpha=0.01)
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
h = model.fit(data["train"], validation_data=data["val"], epochs=cfg.EPOCHS_CNN,
              class_weight=data["class_weight"], verbose=2)
model.save(INFO["path"])

save_run_info(INFO["key"], [h.history], time.time() - t0, model)
report(NAME, INFO["key"], INFO["path"], data)
