from torchvision import transforms
from PIL import Image
import torch

def preprocess_image(img: Image.Image, size: int):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    return transform(img)  # Returns a 3D tensor: [C, H, W]
