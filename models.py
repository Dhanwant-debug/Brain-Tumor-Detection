"""The two model architectures. Both contain augmentation + preprocessing INSIDE the model,
so they accept raw 0-255 pixels and output P(tumor)."""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import config as cfg


@tf.keras.utils.register_keras_serializable(package="brain")
class PerImageStandardize(layers.Layer):
    """Rescales every image to zero mean and unit spread. Removes brightness / contrast differences
    between scanners, which is the biggest single help for the from-scratch CNN."""

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
        return (x - mean) / (std + 1e-3)


def _augmentation(rot, zoom, shift=0.0):
    aug = [layers.RandomFlip("horizontal"), layers.RandomRotation(rot), layers.RandomZoom(zoom)]
    if shift:
        aug.append(layers.RandomTranslation(shift, shift))
    return tf.keras.Sequential(aug, name="augmentation")


def build_mobilenet(img_size=224):
    """Transfer learning: pretrained MobileNetV2 + new classification head.
    Returns (model, base) so the training script can freeze / unfreeze the base."""
    base = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False,
                       weights="imagenet" if cfg.USE_IMAGENET else None)
    base.trainable = False
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = _augmentation(0.1, 0.1)(inputs)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="mobilenetv2_classifier"), base


def build_custom_cnn(img_size=64):
    """Custom CNN trained from scratch (about 155k parameters)."""
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = _augmentation(rot=0.05, zoom=0.1, shift=0.05)(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    x = PerImageStandardize()(x)
    blocks = [(16, 2), (32, 2), (64, 2), (128, 1)]          # (filters, number of conv layers)
    for i, (filters, n) in enumerate(blocks):
        for j in range(n):
            is_last = (i == len(blocks) - 1 and j == n - 1)
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                              kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization(momentum=0.9)(x)   # small data -> fast-updating statistics
            x = layers.Activation("relu", name="last_conv_relu" if is_last else None)(x)  # Grad-CAM target
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="custom_cnn")
