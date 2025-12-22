import cv2
import os
import numpy as np
import albumentations as A
import shutil
import random

# --- CẤU HÌNH ---
INPUT_DIR = './data'            
OUTPUT_DIR = './data_augmented' 
AUGMENT_SIZE = 5                

transform = A.Compose([
    # 1. Xoay nhẹ (-15 đến 15 độ)
    A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
    
    # 2. Dịch chuyển (Shift) và Zoom nhẹ
    A.ShiftScaleRotate(
        shift_limit=0.1,  # Dịch chuyển tối đa 10%
        scale_limit=0.1,  # Zoom to/nhỏ 10%
        rotate_limit=0,   # (Đã xoay ở trên rồi nên chỗ này ko xoay nữa)
        p=0.5, 
        border_mode=cv2.BORDER_CONSTANT, 
        value=0
    ),
    
    # 3. Chỉnh độ sáng (Mô phỏng môi trường sáng/tối)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    
    # 4. Thêm nhiễu hạt nhẹ (Noise) - Giống camera chất lượng thấp
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
])

def augment_data():
    # Dọn dẹp thư mục cũ
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    classes = os.listdir(INPUT_DIR)
    
    total_original = 0
    total_generated = 0

    for class_name in classes:
        class_path = os.path.join(INPUT_DIR, class_name)
        if not os.path.isdir(class_path): continue
            
        print(f"Đang xử lý: {class_name}...")
        
        save_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(save_path, exist_ok=True)
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            
            image = cv2.imread(img_path)
            if image is None: continue
            
            cv2.imwrite(os.path.join(save_path, f"org_{img_name}"), image)
            total_original += 1

            for i in range(AUGMENT_SIZE):
                try:
                    augmented = transform(image=image)['image']
                    new_filename = f"aug_{i}_{img_name}"
                    cv2.imwrite(os.path.join(save_path, new_filename), augmented)
                    total_generated += 1
                except Exception as e:
                    print(f"Lỗi: {e}")

    print(f"\nDữ liệu gốc: {total_original} | Dữ liệu mới: {total_generated}")
    print(f"Tổng cộng: {total_original + total_generated} ảnh trong '{OUTPUT_DIR}'")

if __name__ == "__main__":
    augment_data()