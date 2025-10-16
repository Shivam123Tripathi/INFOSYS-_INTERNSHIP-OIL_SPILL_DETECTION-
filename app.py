# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import jaccard_score, f1_score, accuracy_score

# --- 1) Model loading (TensorFlow Keras .h5 assumed) ---
@st.cache_resource
def load_model(path="C:\shivam_code\INFOSYS-_INTERNSHIP-OIL_SPILL_DETECTION-\oilspill_model.h5"):
    import tensorflow as tf
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        return None
    model = tf.keras.models.load_model(path, compile=False)
    return model

# --- 2) Preprocessing: read uploaded file and transform to model input ---
def preprocess_image(file, target_size=(256,256)):
    # file: upload file-like (BytesIO or UploadedFile)
    image = Image.open(file)
    # convert to RGB if single-channel; adapt if using SAR (single channel) -> you may want .convert("L")
    image_rgb = image.convert("RGB")
    # keep a copy for display at original resolution
    display_image = image_rgb.copy()
    # resize for model input
    image_resized = image_rgb.resize(target_size, Image.BILINEAR)
    arr = np.array(image_resized).astype(np.float32) / 255.0
    # if model expects single channel SAR, you will need to adjust here
    input_arr = np.expand_dims(arr, axis=0)   # shape (1, H, W, C)
    return display_image, input_arr

# --- 3) Postprocess prediction to binary mask ---
def postprocess_mask(prediction, threshold=0.5):
    # prediction: model output shape (1, H, W, 1) or (1, H, W, C)
    pred = np.squeeze(prediction)
    # if model outputs probabilities per pixel, assume single channel
    if pred.ndim == 3 and pred.shape[-1] > 1:
        # if multiple channels, take the first or argmax depending on model
        pred = pred[..., 0]
    mask = (pred > threshold).astype(np.uint8)
    return mask

# --- 4) Overlay mask on original image for visualization ---
def overlay_mask_on_image(display_image_pil, mask_resized):
    img = np.array(display_image_pil).astype(np.uint8)
    # ensure mask_resized is same size as display image
    if mask_resized.shape[:2] != img.shape[:2]:
        mask_resized = cv2.resize(mask_resized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # create colored overlay
    overlay = img.copy()
    overlay[mask_resized == 1] = (255, 0, 0)  # red overlay on mask region
    out = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    return out

# --- 5) Metric computation (if ground truth provided) ---
def compute_metrics(true_mask, pred_mask):
    # flatten
    y_true = true_mask.flatten().astype(np.uint8)
    y_pred = pred_mask.flatten().astype(np.uint8)
    # some metrics require at least one positive example - handle gracefully
    try:
        iou = jaccard_score(y_true, y_pred, average='binary', zero_division=0)
        dice = f1_score(y_true, y_pred, average='binary', zero_division=0)  # Dice == F1 for binary
        acc = accuracy_score(y_true, y_pred)
    except Exception as e:
        iou, dice, acc = 0.0, 0.0, 0.0
    return {"IoU": float(iou), "Dice": float(dice), "Accuracy": float(acc)}

# --- Streamlit UI layout ---
st.set_page_config(layout="wide", page_title="Oil Spill Detection")
st.title("🌊 Oil Spill Detection — upload and analyze")

# Top: file uploader
st.sidebar.header("Upload / Settings")
uploaded_file = st.sidebar.file_uploader("Upload satellite image", type=["jpg","jpeg","png","tif","tiff"])
# optional: allow uploading ground-truth mask for evaluation
gt_file = st.sidebar.file_uploader("(Optional) Upload ground-truth mask (binary image)", type=["png","jpg","tif","tiff"])

threshold = st.sidebar.slider("Prediction threshold", 0.1, 0.9, 0.5, step=0.05)
model_path = st.sidebar.text_input("Model path", value=r"C:\shivam_code\INFOSYS-_INTERNSHIP-OIL_SPILL_DETECTION-\oilspill_model.h5")

# Load model
model = load_model(model_path)

if uploaded_file is None:
    st.info("Upload a satellite image from the sidebar to run detection.")
else:
    # Preprocess
    display_image, input_arr = preprocess_image(uploaded_file, target_size=(256,256))
    st.write("### Original image")
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(display_image, use_column_width=True)

    if model is None:
        st.error("Model not loaded. Ensure oil_spill.h5 is in the project folder and restart.")
    else:
        # Run inference
        with st.spinner("Running model inference..."):
            prediction = model.predict(input_arr)
        pred_mask = postprocess_mask(prediction, threshold=threshold)   # shape (H,W)

        # Resize predicted mask to original display size for overlay/metrics
        display_size = display_image.size  # (width, height)
        pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (display_size[0], display_size[1]), interpolation=cv2.INTER_NEAREST)

        # Show detection result
        with col2:
            st.write("### Predicted mask")
            st.image(pred_mask_resized * 255, clamp=True, channels="GRAY", use_column_width=True)

        # Overlay and show
        st.write("### Overlay (original + predicted mask)")
        overlay = overlay_mask_on_image(display_image, pred_mask_resized)
        st.image(overlay, use_column_width=True)

        # Detection metrics and analysis
        st.write("### Detection metrics & analysis")
        # If user provided a ground-truth mask, compute IoU/Dice/Acc
        if gt_file is not None:
            # preprocess the ground truth mask to binary
            gt_img = Image.open(gt_file).convert("L")
            gt_arr = np.array(gt_img)
            # binarize: consider >127 as positive
            gt_bin = (gt_arr > 127).astype(np.uint8)
            # resize to display size if needed
            if gt_bin.shape[:2] != (display_size[1], display_size[0]):
                gt_bin = cv2.resize(gt_bin, (display_size[0], display_size[1]), interpolation=cv2.INTER_NEAREST)
            metrics = compute_metrics(gt_bin, pred_mask_resized)
            st.json(metrics)
        else:
            # No ground-truth: show estimated spill area & other stats
            area_pixels = int(pred_mask_resized.sum())
            total_pixels = pred_mask_resized.shape[0] * pred_mask_resized.shape[1]
            coverage_pct = (area_pixels / total_pixels) * 100
            st.write(f"Estimated spill coverage: **{coverage_pct:.4f}%**")
            st.write(f"Spill pixel count: **{area_pixels}** of {total_pixels}")

        # Extra: bounding boxes and regions found
        st.write("### Detailed region analysis")
        # find connected components to list regions
        num_labels, labels_im = cv2.connectedComponents(pred_mask_resized.astype(np.uint8))
        regions = []
        for lab in range(1, num_labels):
            mask_lab = (labels_im == lab).astype(np.uint8)
            ys, xs = np.where(mask_lab == 1)
            if ys.size == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            area = mask_lab.sum()
            regions.append({"label": lab, "area_pixels": int(area), "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]})
        st.write(f"Found **{len(regions)}** connected region(s).")
        if regions:
            st.table(regions)