import os
import cv2

# Source and destination folders
source_folder = r"D:\data sets"
dest_folder = r"D:\forgery_project_finalxxxxx\dataset\real"

# Create destination folder if it doesn't exist
os.makedirs(dest_folder, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Resize and save
for idx, file in enumerate(image_files):
    src_path = os.path.join(source_folder, file)
    img = cv2.imread(src_path)

    if img is not None:
        resized = cv2.resize(img, (256, 256))
        dest_path = os.path.join(dest_folder, f"real_{idx}.jpg")
        cv2.imwrite(dest_path, resized)

print(f"✅ {len(image_files)} images resized and saved to {dest_folder}")
