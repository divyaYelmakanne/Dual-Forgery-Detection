import torch
import torch.nn as nn
from torchvision import models
from model import SimpleCNN

def load_model(path, model_type):
    if model_type == "cnn":
        model = SimpleCNN()
    elif model_type == "resnet":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def predict(model, image_tensor):
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        return prediction.item(), confidence.item()