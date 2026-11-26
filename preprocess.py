"""Image preprocessing shared by training and the web app."""
import numpy as np
from PIL import Image


def crop_brain(a, thr=25):
    """Crop an RGB uint8 array to the bounding box of the (bright) brain, removing black borders."""
    mask = a.mean(axis=2) > thr
    rows = np.where(mask.sum(axis=1) > 3)[0]
    cols = np.where(mask.sum(axis=0) > 3)[0]
    if len(rows) < 10 or len(cols) < 10:
        return a
    c = a[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return c if c.shape[0] * c.shape[1] > 0.25 * a.shape[0] * a.shape[1] else a


def prepare(pil_img, size, crop):
    """PIL image -> uint8 array (size, size, 3), optionally brain-cropped first."""
    a = np.array(pil_img.convert("RGB"))
    if crop:
        a = crop_brain(a)
    return np.array(Image.fromarray(a).resize((size, size), Image.BICUBIC), dtype="uint8")
