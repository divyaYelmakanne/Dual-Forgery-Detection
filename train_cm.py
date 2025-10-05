import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import cv2
import shutil
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

# Paths
# --- Assuming your cm1 folder has 'fake' and 'real' subfolders ---
data_dir = r"D:\forgery_project_finalxxxxx\dataset\cm\cm1"
processed_data_dir = f"{data_dir}_prepared"
save_dir = "okay"

# --- Define functions that can be called by child processes ---
def prepare_data(data_source, dest, fake_subdir='fake'):
    print("--- Preparing data for full-image classification ---")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    
    os.makedirs(os.path.join(dest, 'train', 'real'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'train', 'fake'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'val', 'real'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'val', 'fake'), exist_ok=True)

    all_real = [f for f in os.listdir(os.path.join(data_source, 'real')) if f.endswith(('.jpg', '.png', '.JPG'))]
    all_fake = [f for f in os.listdir(os.path.join(data_source, fake_subdir)) if f.endswith(('.jpg', '.png', '.JPG'))]
    random.shuffle(all_real)
    random.shuffle(all_fake)

    num_train_real = int(len(all_real) * 0.8)
    num_train_fake = int(len(all_fake) * 0.8)
    
    print(f"Moving {num_train_real} real images and {num_train_fake} fake images to training sets...")
    for fname in all_real[:num_train_real]:
        shutil.copy(os.path.join(data_source, 'real', fname), os.path.join(dest, 'train', 'real', fname))
    for fname in all_fake[:num_train_fake]:
        shutil.copy(os.path.join(data_source, fake_subdir, fname), os.path.join(dest, 'train', 'fake', fname))

    print(f"Moving {len(all_real) - num_train_real} real images and {len(all_fake) - num_train_fake} fake images to validation sets...")
    for fname in all_real[num_train_real:]:
        shutil.copy(os.path.join(data_source, 'real', fname), os.path.join(dest, 'val', 'real', fname))
    for fname in all_fake[num_train_fake:]:
        shutil.copy(os.path.join(data_source, fake_subdir, fname), os.path.join(dest, 'val', 'fake', fname))
    print("Data preparation complete.")

def generate_gradcam(model, image_tensor, class_idx=None):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(class_idx or torch.argmax(model(image_tensor)).item())])
    grayscale_cam = grayscale_cam[0, :]
    
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image = np.float32(image)
    
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    return cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

# --- START OF MAIN EXECUTION ---
if __name__ == '__main__':
    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NEW: Delete previous results to ensure a fresh run
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    os.makedirs(f"{save_dir}/gradcam", exist_ok=True)
    os.makedirs(f"{save_dir}/metrics", exist_ok=True)
    os.makedirs(f"{save_dir}/model", exist_ok=True)

    # --- CORRECT DATA PREPARATION: Reorganize folders for proper split ---
    # MODIFICATION: Pass the name of the subdirectory for fake images
    prepare_data(data_dir, processed_data_dir, fake_subdir='fffake')

    # --- SEPARATE TRANSFORMS ---
    # MODIFICATION 1: ADD MORE AGGRESSIVE DATA AUGMENTATION
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = ImageFolder(root=f"{processed_data_dir}/train", transform=train_transform)
    val_dataset = ImageFolder(root=f"{processed_data_dir}/val", transform=val_transform)
    class_names = train_dataset.classes

    # --- Oversampling the minority class ---
    class_counts = [0, 0]
    for _, label in train_dataset:
        class_counts[label] += 1

    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_dataset.targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Model
    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), 
        nn.Linear(num_ftrs, 2)
    )
    model = model.to(device)

    # Loss and Optimizer
    # MODIFICATION 2: USE CLASS-WEIGHTED LOSS
    class_weights_tensor = torch.tensor(
        [1.0, class_counts[0] / class_counts[1]],  # Weight fake class more
        dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Training
    train_acc_list, val_acc_list = [], []
    best_val_acc = 0
    epochs = 10
    for epoch in range(epochs):
        model.train()
        correct, total, total_loss = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        train_acc = correct / total
        train_acc_list.append(train_acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/model/best_model.pth")
            print("Best model saved.")

    # Accuracy Plot
    plt.figure()
    plt.plot(train_acc_list, label='Train Acc')
    plt.plot(val_acc_list, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train vs Val Accuracy')
    plt.savefig(f"{save_dir}/plots/accuracy_plot.png")
    plt.close()

    # Confusion Matrix
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"{save_dir}/metrics/confusion_matrix.png")
    plt.close()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Apply Grad-CAM on few validation images
    print("\nGenerating Grad-CAM heatmaps for sample images...")
    count = 0
    for inputs, labels in val_loader:
        for i in range(inputs.size(0)):
            image_tensor = inputs[i]
            heatmap_image = generate_gradcam(model, image_tensor)
            save_path = os.path.join(save_dir, "gradcam", f"gradcam_{count}.jpg")
            cv2.imwrite(save_path, heatmap_image)
            count += 1
            if count >= 20:
                break
        if count >= 20:
            break
    print("Grad-CAMs saved.")