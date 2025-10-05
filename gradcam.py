# utils/gradcam.py
import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_heatmap(model, image_tensor, model_type='resnet'):
    model.eval()

    # Fix input shape if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.dim() == 5:
        image_tensor = image_tensor.squeeze(0)

    # Choose the right target layer
    if model_type == 'resnet':
        target_layers = [model.layer4[-1]]
    elif model_type == 'cnn':
        target_layers = [model.features[3]]  # Last conv layer before flatten
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(1)])[0]

    img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    visualization = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
    return visualization