# utils.py
import torch
import numpy as np
import cv2

def generate_gradcam(img_tensor, model, target_class):
    img_tensor.requires_grad_()
    model.eval()

    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    feature_maps, gradients = None, None
    target_layer = model.features[-3]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    score = output[0, target_class]
    model.zero_grad()
    score.backward()

    h1.remove(); h2.remove()
    pooled_grad = torch.mean(gradients, dim=[0, 2, 3])
    feature_maps = feature_maps[0]
    for i in range(feature_maps.shape[0]):
        feature_maps[i] *= pooled_grad[i]
    heatmap = feature_maps.mean(0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap
