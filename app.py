"""Streamlit app: run one or both models on an MRI and compare predictions + Grad-CAM.
Run:  streamlit run app.py   (after training)"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

import config as cfg
import models  # noqa: F401  (registers the custom layer so the CNN can be loaded)
from preprocess import prepare
from gradcam import make_gradcam, overlay_heatmap

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor Detection: model comparison")
st.caption("Academic project. Not a medical diagnostic tool.")


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


available = [n for n, m in cfg.MODELS.items() if os.path.exists(m["path"])]
if not available:
    st.error("No trained models found. Run `python train_mobilenet.py` and `python train_cnn.py` first.")
    st.stop()

selected = st.sidebar.multiselect("Models to run", available, default=available)
st.sidebar.caption("Select both to compare predictions side by side.")

tab_predict, tab_compare = st.tabs(["Predict", "Model comparison"])

with tab_predict:
    file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])
    if file and selected:
        img = Image.open(file).convert("RGB")

        cols = st.columns(1 + len(selected))
        cols[0].image(img, caption="Uploaded MRI", width="stretch")
        verdicts = {}
        for col, name in zip(cols[1:], selected):
            m = cfg.MODELS[name]
            arr_u8 = prepare(img, m["img_size"], m["crop"])           # same preprocessing as training
            arr = np.expand_dims(arr_u8.astype("float32"), 0)
            model = load_model(m["path"])
            cam, prob = make_gradcam(model, arr)
            note = "brain-cropped, " if m["crop"] else ""
            col.image(overlay_heatmap(arr_u8, cam), caption=f"{name}: Grad-CAM ({note}{m['img_size']}px input)",
                      width="stretch")
            verdicts[name] = prob > 0.5
            if prob > 0.5:
                col.error(f"{name}: Tumor detected ({prob:.1%})")
            else:
                col.success(f"{name}: No tumor detected ({1 - prob:.1%})")
            col.progress(prob, text=f"Tumor probability: {prob:.1%}")

        if len(set(verdicts.values())) > 1:
            st.warning("The models disagree on this image. Treat the result as uncertain.")
        st.info("Red/yellow areas contributed most to each model's tumor score. "
                "The heatmap shows what the model focused on, not a confirmed tumor location.")

with tab_compare:
    csv = f"{cfg.RESULTS_DIR}/comparison.csv"
    if not os.path.exists(csv):
        st.info("Run `python compare.py` to generate the comparison results.")
    else:
        st.subheader("Test-set metrics")
        st.dataframe(pd.read_csv(csv).set_index("Model").T, width="stretch")
        for img_file, cap in [("comparison_metrics.png", "Metrics"), ("comparison_confusion.png", "Confusion matrices"),
                              ("comparison_roc.png", "ROC curves"), ("comparison_training.png", "Training curves")]:
            p = f"{cfg.RESULTS_DIR}/{img_file}"
            if os.path.exists(p):
                st.image(p, caption=cap)
