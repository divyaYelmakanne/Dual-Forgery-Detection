import os, cv2, numpy as np
from glob import glob


REAL_DIR = '../dataset/real'
MASK_DIR = '../dataset/masks'
MASKED_DIR = '../dataset/masked'



os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(MASKED_DIR, exist_ok=True)

def generate_random_mask(img_size=(256, 256), box_size=(64, 64)):
    mask = np.zeros(img_size, dtype=np.uint8)
    y = np.random.randint(0, img_size[0] - box_size[0])
    x = np.random.randint(0, img_size[1] - box_size[1])
    mask[y:y+box_size[0], x:x+box_size[1]] = 255
    return mask

image_paths = glob(f"{REAL_DIR}/*.jpg")
for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    mask = generate_random_mask()
    masked = cv2.bitwise_and(img, img, mask=255-mask)

    cv2.imwrite(f"{MASK_DIR}/mask_{i}.png", mask)
    cv2.imwrite(f"{MASKED_DIR}/masked_{i}.png", masked)
