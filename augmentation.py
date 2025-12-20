import cv2
import numpy as np
import os
import random

def augment_image(img):
    h, w = img.shape[:2]

    # ----- Rotation nhẹ -----
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # ----- Scale -----
    scale = random.uniform(0.85, 1.15)
    img = cv2.resize(img, None, fx=scale, fy=scale)

    # Pad hoặc crop về size gốc
    img = cv2.resize(img, (w, h))

    # ----- Translation -----
    tx = random.randint(-15, 15)
    ty = random.randint(-15, 15)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

   # ----- Brightness / Contrast -----
    alpha = random.uniform(0.9, 1.1)
    beta  = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img


def augment_dataset(input_dir, output_dir, n_aug=3):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue

        img = cv2.imread(os.path.join(input_dir, fname))
        cv2.imwrite(os.path.join(output_dir, fname), img)

        for i in range(n_aug):
            aug = augment_image(img)
            name = f"{os.path.splitext(fname)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(output_dir, name), aug)

input_dir = 'data/'
output_dir = 'data_Aug/'
# augment_dataset(input_dir, output_dir, n_aug=3)

for cls in os.listdir(input_dir):
    augment_dataset(
        input_dir=os.path.join(input_dir, cls),
        output_dir=os.path.join(output_dir, cls),
        n_aug=3  
    )