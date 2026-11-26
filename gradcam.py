"""Grad-CAM that works for BOTH models.
- MobileNetV2 is nested inside the model, so we split it into conv part + head.
- The custom CNN is flat, so we tap its layer named 'last_conv_relu'."""
import numpy as np
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

DISPLAY = (224, 224)


def _mobilenet_base(model):
    return next((l for l in model.layers if l.name.startswith("mobilenetv2")), None)


def _finish(grads, conv_out):
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))                  # importance of each channel
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def make_gradcam(model, img_array):
    """img_array: float32 (1, H, W, 3), raw pixels 0-255, already preprocessed for this model.
    Returns (heatmap in [0,1], probability of tumor)."""
    base = _mobilenet_base(model)
    if base is not None:                                             # MobileNetV2 model
        conv_model = tf.keras.Model(base.input, base.get_layer("out_relu").output)
        head = model.layers[model.layers.index(base) + 1:]
        x = preprocess_input(img_array.copy())
        with tf.GradientTape() as tape:
            conv_out = conv_model(x, training=False)
            tape.watch(conv_out)
            y = conv_out
            for layer in head:
                y = layer(y, training=False)
            prob = y[:, 0]
    else:                                                            # custom CNN
        grad_model = tf.keras.Model(model.input, [model.get_layer("last_conv_relu").output, model.output])
        with tf.GradientTape() as tape:
            conv_out, pred = grad_model(tf.convert_to_tensor(img_array), training=False)
            prob = pred[:, 0]
    grads = tape.gradient(prob, conv_out)
    return _finish(grads, conv_out), float(prob.numpy()[0])


def overlay_heatmap(img_uint8, cam, alpha=0.4):
    """Blend the heatmap over the image the MODEL saw (uint8 array); returns a PIL image."""
    base = np.array(Image.fromarray(img_uint8).resize(DISPLAY, Image.BICUBIC), dtype="float32") / 255.0
    heat = np.array(Image.fromarray(np.uint8(cam * 255)).resize(DISPLAY, Image.BICUBIC), dtype="float32") / 255.0
    colored = colormaps["jet"](heat)[..., :3]
    return Image.fromarray(np.uint8(np.clip((1 - alpha) * base + alpha * colored, 0, 1) * 255))
