import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from train import SimpleCNN
import matplotlib.pyplot as plt
import os

# Load the trained model
model = SimpleCNN()
model.load_state_dict(torch.load("forgery_classifier.pth", map_location='cpu'))
model.eval()

# Hook setup
gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

# Register hooks on last conv layer
model.features[-3].register_forward_hook(forward_hook)
model.features[-3].register_backward_hook(backward_hook)

# Image path to test
img_path = "dataset/fake/masked_0.png"  # change if needed
img_name = os.path.basename(img_path).split('.')[0]

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# Forward + backward
output = model(input_tensor)
class_idx = output.argmax().item()
model.zero_grad()
output[0, class_idx].backward()

# Grad-CAM calculation
grad = gradients[0].squeeze().detach().numpy()
act = activations[0].squeeze().detach().numpy()
weights = grad.mean(axis=(1, 2))
cam = np.zeros(act.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * act[i]
cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (256, 256))
cam = cam / cam.max()

# Overlay on original image
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
img_np = np.array(img.resize((256, 256)))
overlay = heatmap * 0.4 + img_np * 0.6

# Save
os.makedirs("gradcam_outputs", exist_ok=True)
cv2.imwrite(f"gradcam_outputs/{img_name}_gradcam.jpg", overlay[:, :, ::-1])

# Show
plt.imshow(overlay[:, :, ::-1].astype(np.uint8))
plt.title(f"Grad-CAM - Class {class_idx}")
plt.axis('off')
plt.tight_layout()
plt.show()
