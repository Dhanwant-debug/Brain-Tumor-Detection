# Brain Tumor Classification: MobileNetV2 vs Custom CNN

Binary classification (tumor / no tumor) of brain MRI images with two models trained on the same data split.

## Project structure
| File | Purpose |
|---|---|
| `config.py` | Shared settings (paths, image size, epochs, seed) |
| `data.py` | Loads images into memory, one stratified 70/15/15 split used by both models |
| `preprocess.py` | Brain cropping and resizing (shared by training and the app) |
| `models.py` | Architectures: MobileNetV2 (transfer learning) and custom CNN |
| `train_mobilenet.py` | Trains and evaluates model 1 |
| `train_cnn.py` | Trains and evaluates model 2 |
| `evaluate.py` | Metrics and confusion matrix helpers |
| `compare.py` | Compares both models: table + 4 plots |
| `gradcam.py` | Grad-CAM heatmaps for both models |
| `app.py` | Streamlit web app (predict with one or both models, view comparison) |
| `cv_cnn.py` | 5-fold cross-validation of the custom CNN (more reliable than one small test set) |
| `utils.py` | Saves / loads training history, time, parameter count |

## Setup (Python 3.12 recommended)
```
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Place the Kaggle dataset as `dataset/yes` and `dataset/no`
(https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection).

## Run, in this order
```
python train_mobilenet.py     # saves saved_models/mobilenetv2_model.keras
python train_cnn.py           # saves saved_models/custom_cnn_model.keras
python compare.py             # writes results/comparison.csv and plots
python cv_cnn.py              # optional: 5-fold cross-validation of the custom CNN (about 15 min)
streamlit run app.py          # web app
```
Outputs go to `saved_models/` and `results/`.

## Already have a trained MobileNetV2 model?
Skip `train_mobilenet.py`. Copy your existing model file into this project as
`saved_models/mobilenetv2_model.keras`, then run only:
```
python train_cnn.py
python compare.py
```
The comparison is fair only if your existing model was trained with the stratified 70/15/15 split (seed 42),
i.e. the corrected `train.py`, so both models are tested on the same 38 images.

## Custom CNN recipe (why it works)
Trained from scratch on ~177 images, the CNN first collapsed to predicting "tumor" for every image (about 60%).
What fixed it: brain crop + small 64x64 input, per-image standardization, fixed-length training (120 epochs) with
cosine learning-rate decay and the FINAL weights kept (restoring the best-validation-loss epoch picked an early,
collapsed model), BatchNorm momentum 0.9, L2 1e-4, dropout 0.3, light augmentation and class weights.
On a 38-image test set results swing a lot (about 70-90% depending on the split), so use `cv_cnn.py` for a stable estimate.
