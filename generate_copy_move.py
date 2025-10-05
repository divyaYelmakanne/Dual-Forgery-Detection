import os
import cv2
import random
from PIL import Image, ImageChops
from glob import glob

# Define the paths for your real and fake images using a raw string (r"...")
REAL_DIR = r'D:\forgery_project_finalxxxxx\dataset\cm\cm1\real'
FAKE_CM_DIR = r'D:\forgery_project_finalxxxxx\dataset\cm\cm1\fffake'

# Create the new directory for copy-move forgeries if it doesn't exist
os.makedirs(FAKE_CM_DIR, exist_ok=True)

def create_copy_move_forgery(image_path, output_path):
    """
    Creates a copy-move forgery from a given image.
    A patch from the image is copied, transformed, and pasted back.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        # Define patch size (e.g., 1/4 of the image dimensions)
        patch_size = (width // 4, height // 4)
        
        # Randomly select a patch to copy
        x1 = random.randint(0, width - patch_size[0])
        y1 = random.randint(0, height - patch_size[1])
        patch = img.crop((x1, y1, x1 + patch_size[0], y1 + patch_size[1]))

        # Apply random transformations to the patch using correct Image methods
        transformations = [
            patch,
            patch.transpose(Image.FLIP_LEFT_RIGHT), # Correct method for mirroring
            patch.transpose(Image.FLIP_TOP_BOTTOM), # Correct method for flipping
            patch.rotate(random.choice([90, 180, 270])),
        ]
        transformed_patch = random.choice(transformations)

        # Randomly select a new location to paste the patch
        x2 = random.randint(0, width - patch_size[0])
        y2 = random.randint(0, height - patch_size[1])
        
        # Ensure the new location is not too close to the original
        while abs(x1 - x2) < patch_size[0] or abs(y1 - y2) < patch_size[1]:
            x2 = random.randint(0, width - patch_size[0])
            y2 = random.randint(0, height - patch_size[1])

        # Create a copy to paste the patch on
        new_img = img.copy()
        new_img.paste(transformed_patch, (x2, y2))
        new_img.save(output_path)
        return True

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

# Get all real image paths
image_paths = glob(os.path.join(REAL_DIR, '*.jpg')) + glob(os.path.join(REAL_DIR, '*.png'))

print(f"Found {len(image_paths)} real images to create forgeries from.")

# Generate forgeries and save them
for i, path in enumerate(image_paths):
    output_path = os.path.join(FAKE_CM_DIR, f"copy_move_{i}.jpg")
    create_copy_move_forgery(path, output_path)
    print(f"Generated forgery {i+1}/{len(image_paths)}: {os.path.basename(output_path)}")

print("\nCopy-move forgery generation complete.")