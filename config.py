"""Shared settings for the whole project. Change values here, not in every script."""
import os

DATA_DIR = os.environ.get("DATA_DIR", "dataset")   # must contain 'yes' and 'no' folders
BATCH = 16
SEED = 42

MODEL_DIR = "saved_models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Each model has its own input size / preprocessing, but BOTH use the same train/val/test split.
MODELS = {
    "MobileNetV2": {"key": "mobilenetv2", "path": f"{MODEL_DIR}/mobilenetv2_model.keras",
                    "img_size": 224, "crop": False},
    "Custom CNN": {"key": "custom_cnn", "path": f"{MODEL_DIR}/custom_cnn_model.keras",
                   "img_size": 64, "crop": True},      # brain-cropped, small input
}

EPOCHS_HEAD = int(os.environ.get("EPOCHS_HEAD", 20))   # MobileNetV2 phase 1
EPOCHS_FINE = int(os.environ.get("EPOCHS_FINE", 10))   # MobileNetV2 phase 2 (fine-tuning)
EPOCHS_CNN = int(os.environ.get("EPOCHS_CNN", 120))    # custom CNN: fixed length, cosine LR decay
USE_IMAGENET = os.environ.get("USE_IMAGENET", "1") == "1"

CLASS_NAMES = ["No Tumor", "Tumor"]
