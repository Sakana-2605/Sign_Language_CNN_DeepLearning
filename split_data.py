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

# 1. Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, # Mode ảnh tĩnh để đạt độ chính xác cao nhất khi trích xuất
    max_num_hands=1,
    min_detection_confidence=0.5
)

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    # 1. Tịnh tiến: Trừ mọi điểm cho tọa độ cổ tay
    base_node = landmarks[0]
    landmarks = landmarks - base_node
    
    # 2. Scale: Chia cho giá trị tuyệt đối lớn nhất
    max_value = np.abs(landmarks).max()
    if max_value > 0:
        landmarks = landmarks / max_value
        
    return landmarks.flatten() # Trả về mảng 1D 63 phần tử

def process_dataset(input_base_dir):
    X_data = []
    y_labels = []

    # Liệt kê các thư mục con (tương ứng với các nhãn S, T, U, V, X, Y)
    classes = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
    print(f"Tìm thấy các lớp: {classes}")

    for label in classes:
        class_dir = os.path.join(input_base_dir, label)
        print(f"Đang xử lý lớp: {label}...")
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Chuyển BGR sang RGB cho MediaPipe
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_img)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Lấy 21 điểm x, y, z
                    raw_lms = []
                    for lm in hand_lms.landmark:
                        raw_lms.extend([lm.x, lm.y, lm.z])
                    
                    # CHUẨN HÓA
                    normalized_lms = normalize_landmarks(raw_lms)
                    
                    X_data.append(normalized_lms)
                    y_labels.append(label)
            else:
                print(f"  [!] Không tìm thấy tay trong ảnh: {img_name}")

    # Chuyển sang Numpy array
    X_data = np.array(X_data, dtype=np.float32)
    y_labels = np.array(y_labels)

    # LƯU FILE .NPY
    np.save('X_data.npy', X_data)
    np.save('y_labels.npy', y_labels)
    
    print("-" * 30)
    print(f"Hoàn tất! Đã lưu {len(X_data)} mẫu dữ liệu.")
    print(f"Shape của X: {X_data.shape} (Số mẫu, 63 tọa độ)")
    print(f"Shape của y: {y_labels.shape}")

# Chạy script (thay 'data_cnn/train' bằng đường dẫn thư mục ảnh của bạn)
if __name__ == "__main__":
    input_folder = 'data/' 
    if os.path.exists(input_folder):
        process_dataset(input_folder)
    else:
        print(f"Lỗi: Thư mục {input_folder} không tồn tại.")