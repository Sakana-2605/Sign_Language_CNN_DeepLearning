# import os
# import shutil
# import random

# input_dir = "data_pipeline"
# ouput_dir = "data_cnn"

# # input_dir = "data_cnn"
# # ouput_dir = "data_cnn_train"
# SPLIT_RATIO = 0.8
# SEED = 42

# random.seed(SEED)

# #create folders
# for split in ["train", "val"]:
#     for cls in os.listdir(input_dir):
#         os.makedirs(os.path.join(ouput_dir, split, cls), exist_ok=True)

# # split dataset
# for cls in os.listdir(input_dir):
#     imgs = os.listdir(os.path.join(input_dir, cls))
#     random.shuffle(imgs)

#     split_idx = int(len(imgs) * SPLIT_RATIO)
#     train_imgs = imgs[:split_idx]
#     val_imgs   = imgs[split_idx:]

#     for img in train_imgs:
#         shutil.copy(
#             os.path.join(input_dir, cls, img),
#             os.path.join(ouput_dir, "train", cls, img)
#         )

#     for img in val_imgs:
#         shutil.copy(
#             os.path.join(input_dir, cls, img),
#             os.path.join(ouput_dir, "val", cls, img)
#         )

# print("Done")

import cv2
import mediapipe as mp
import numpy as np
import os

# 1. Khởi tạo MediaPipe Hands với cấu hình tối ưu hơn
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1,
    min_detection_confidence=0.3, # Giảm ngưỡng để bắt được nhiều tay hơn
    model_complexity=1           # Tăng độ phức tạp để tăng độ chính xác (0 hoặc 1)
)

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base_node = landmarks[0]
    landmarks = landmarks - base_node
    
    max_value = np.abs(landmarks).max()
    if max_value > 0:
        landmarks = landmarks / max_value
        
    return landmarks.flatten()

def process_dataset(input_base_dir):
    X_data = []
    y_labels = []

    classes = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
    print(f"Tìm thấy các lớp: {classes}")

    for label in classes:
        class_dir = os.path.join(input_base_dir, label)
        print(f"Đang xử lý lớp: {label}...")
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # --- BƯỚC TỐI ƯU: Tiền xử lý ảnh ---
            # 1. Resize ảnh nếu quá lớn (MediaPipe hoạt động tốt nhất ở size vừa phải)
            h, w = img.shape[:2]
            if h > 1000 or w > 1000:
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

            # 2. Chuyển sang RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Thử nhận diện lần 1
            results = hands.process(rgb_img)

            # 3. Nếu lần 1 thất bại, thử tăng độ tương phản (CLAHE)
            if not results.multi_hand_landmarks:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl,a,b))
                enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                results = hands.process(enhanced_img)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    raw_lms = []
                    for lm in hand_lms.landmark:
                        raw_lms.extend([lm.x, lm.y, lm.z])
                    
                    normalized_lms = normalize_landmarks(raw_lms)
                    X_data.append(normalized_lms)
                    y_labels.append(label)
            else:
                # Ghi lại những file bị lỗi để bạn kiểm tra thủ công sau
                print(f"  [!] Thất bại hoàn toàn: {img_name}")

    X_data = np.array(X_data, dtype=np.float32)
    y_labels = np.array(y_labels)

    np.save('X_data.npy', X_data)
    np.save('y_labels.npy', y_labels)
    
    print("-" * 30)
    print(f"Hoàn tất! Đã lưu {len(X_data)} mẫu dữ liệu.")

if __name__ == "__main__":
    input_folder = 'data/' 
    if os.path.exists(input_folder):
        process_dataset(input_folder)