import streamlit as st
from PIL import Image
import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from preprocess import preprocess_image
from predictor import load_model, predict
from gradcam import generate_heatmap

st.title("🕵️ Image Forgery Detection App")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    inpainting_model_path = "models/forgery_classifier.pth"
    copy_move_model_path = "models/best_model.pth"

    try:
        # Pass the model_type explicitly
        inpainting_model = load_model(inpainting_model_path, model_type="cnn")
        copy_move_model = load_model(copy_move_model_path, model_type="resnet")

        inpaint_tensor = preprocess_image(img, size=256)
        copy_move_tensor = preprocess_image(img, size=224)

        inpaint_label, inpaint_conf = predict(inpainting_model, inpaint_tensor)
        copy_move_label, copy_move_conf = predict(copy_move_model, copy_move_tensor)

        st.subheader("🖌️ Inpainting Forgery Detection")
        st.write("Prediction: {} ({:.2f} confidence)".format(
            "🟥 Fake" if inpaint_label else "✅ Real", inpaint_conf))
        if inpaint_label == 1:
            heatmap = generate_heatmap(inpainting_model, inpaint_tensor, model_type="cnn")
            st.image(heatmap, caption="🧯 Inpainting Heatmap", use_container_width=True)

        st.subheader("🧩 Copy-Move Forgery Detection")
        st.write("Prediction: {} ({:.2f} confidence)".format(
            "🟥 Fake" if copy_move_label else "✅ Real", copy_move_conf))
        if copy_move_label == 1:
            heatmap = generate_heatmap(copy_move_model, copy_move_tensor, model_type="resnet")
            st.image(heatmap, caption="🧯 Copy-Move Heatmap", use_container_width=True)

    except Exception as e:
        st.error("❌ An error occurred during detection: {}".format(e))