# app.py (Streamlit app with optional Google Drive download)
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model

# Optional: gdown for Google Drive download
try:
    import gdown
except Exception:
    gdown = None

# ---------------------------
# simple dice/iou if needed
# ---------------------------
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def iou_keras(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + (1 - dice_coef(y_true, y_pred))


# load model (with fallback download)

MODEL_LOCAL = "oilspill_model.h5"

@st.cache_resource
def get_model():
    # if model exists locally, load it
    if os.path.exists(MODEL_LOCAL):
        try:
            return load_model(MODEL_LOCAL, custom_objects={'bce_dice_loss': bce_dice_loss, 'iou_keras': iou_keras, 'dice_coef': dice_coef})
        except Exception:
            return load_model(MODEL_LOCAL, compile=False)

    # otherwise try to download from Google Drive if SECRET provided
    drive_id = None
    # Streamlit secrets: put your drive file id as MODEL_DRIVE_ID
    try:
        drive_id = st.secrets["MODEL_DRIVE_ID"]
    except Exception:
        drive_id = None

    if drive_id and gdown is not None:
        url = f"https://drive.google.com/uc?id={drive_id}"
        try:
            gdown.download(url, MODEL_LOCAL, quiet=False)
            return load_model(MODEL_LOCAL, custom_objects={'bce_dice_loss': bce_dice_loss, 'iou_keras': iou_keras, 'dice_coef': dice_coef})
        except Exception:
            return load_model(MODEL_LOCAL, compile=False)
    else:
        return None

# ---------------------------
# helpers
# ---------------------------
def preprocess_pil_image(pil_img, target_size=(256,256)):
    img = np.array(pil_img.convert("RGB"))
    img_resized = cv2.resize(img, target_size)
    img_norm = img_resized.astype(np.float32) / 255.0
    return img_norm

def mask_to_overlay(mask, color=(255,0,0)):
    overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    overlay[mask==1] = color
    return overlay

def add_alpha_overlay(image_rgb, overlay_rgb, alpha=0.5):
    blended = cv2.addWeighted(image_rgb.astype(np.uint8), 1-alpha, overlay_rgb.astype(np.uint8), alpha, 0)
    return blended

def image_to_bytes(img_array):
    _, im_buf = cv2.imencode('.png', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    return im_buf.tobytes()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI SpillGuard - Oil Spill Detection", layout="centered")
st.title("🌊 AI SpillGuard — Oil Spill Detection")

st.sidebar.header("Settings")
model_path_input = st.sidebar.text_input("Model path (leave as default unless custom):", value=MODEL_LOCAL)
threshold = st.sidebar.slider("Mask threshold", 0.1, 0.9, 0.5, 0.05)
save_outputs = st.sidebar.checkbox("Save output image to server", value=False)

# load model
model = get_model()
if model is None:
    st.error("Model not found. Add oilspill_model.h5 to the repo or set MODEL_DRIVE_ID in Streamlit secrets pointing to a Google Drive file.")
    st.stop()

uploaded_file = st.file_uploader("Upload satellite image", type=["jpg","jpeg","png","tif"])
if uploaded_file is None:
    st.info("Upload an image to run detection.")
    st.stop()

image = Image.open(uploaded_file)
st.image(image, caption="Uploaded image", use_column_width=True)

img_pre = preprocess_pil_image(image, target_size=(256,256))
inp = np.expand_dims(img_pre, axis=0)

with st.spinner("Running model..."):
    pred = model.predict(inp)[0]
    if pred.ndim==3 and pred.shape[-1]>1:
        pred = pred[...,0]
    pred_prob = pred
    pred_mask = (pred_prob >= threshold).astype(np.uint8)

area_percent = pred_mask.sum() / pred_mask.size * 100.0
overlay_rgb = mask_to_overlay(pred_mask, color=(255,0,0))
overlay_result = add_alpha_overlay((img_pre*255).astype(np.uint8), overlay_rgb, alpha=0.5)

st.subheader("Results")
c1, c2, c3 = st.columns(3)
c1.image((img_pre*255).astype(np.uint8), caption="Resized input")
c2.image((pred_prob*255).astype(np.uint8), caption="Predicted probability")
c3.image((pred_mask*255).astype(np.uint8), caption="Binary mask")

st.image(overlay_result, caption="Overlay (red = detected)")

st.metric("Area detected (%)", f"{area_percent:.2f}%")
if save_outputs:
    os.makedirs("outputs", exist_ok=True)
    fn = f"outputs/result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    with open(fn, "wb") as f:
        f.write(image_to_bytes(overlay_result))
    st.success(f"Saved {fn}")

st.download_button("Download overlay (PNG)", data=image_to_bytes(overlay_result), file_name="oil_spill_overlay.png", mime="image/png")

