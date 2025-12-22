import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import copy
import itertools
import time

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- CẤU HÌNH ---
MODEL_CNN_PATH = 'models/sign_language_model.h5'
MODEL_LANDMARK_PATH = 'models/keypoint_classifier.h5'
CLASSES = ['A', 'B', 'None'] 

# TỐI ƯU TỐC ĐỘ: Chỉ predict mỗi 5 frame (skip 4 frame)
SKIP_FRAMES = 5 

# TRỌNG SỐ (Điều chỉnh lại chút để Landmark gánh nhiều hơn vì nó nhẹ và chuẩn hơn)
WEIGHT_LANDMARK = 0.7
WEIGHT_CNN = 0.3
CONFIDENCE_THRESHOLD = 0.7

# --- LOAD MODELS ---
print("⏳ Đang load models...")
model_cnn = tf.keras.models.load_model(MODEL_CNN_PATH)
model_landmark = tf.keras.models.load_model(MODEL_LANDMARK_PATH)
print("✅ Sẵn sàng!")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- HÀM HỖ TRỢ ---
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    flatten_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, flatten_list)))
    def normalize_(n): return n / max_value if max_value != 0 else 0
    flatten_list = list(map(normalize_, flatten_list))
    return flatten_list

def get_bbox(landmark_list, h, w, padding=20):
    pts = np.array(landmark_list)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    box_w, box_h = x_max - x_min, y_max - y_min
    max_side = max(box_w, box_h) + padding * 2
    center_x, center_y = x_min + box_w // 2, y_min + box_h // 2
    x1, y1 = center_x - max_side // 2, center_y - max_side // 2
    return int(x1), int(y1), int(x1 + max_side), int(y1 + max_side), int(max_side)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
frame_count = 0

# Biến lưu kết quả cũ để hiển thị trong lúc skip frame
current_label = "..."
current_color = (0, 0, 255)
current_confidence = 0.0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    debug_image = copy.deepcopy(frame)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Vẽ khung xương luôn cho ngầu
            mp.solutions.drawing_utils.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- LOGIC SKIP FRAME ---
            frame_count += 1
            if frame_count % SKIP_FRAMES != 0:
                # Nếu đang trong giai đoạn skip, vẽ lại kết quả cũ và bỏ qua tính toán
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                x1, y1, x2, y2, _ = get_bbox(landmark_list, h, w, padding=30)
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), current_color, 2)
                cv2.putText(debug_image, f"{current_label} ({current_confidence*100:.0f}%)", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)
                cv2.imshow('Final System', debug_image)
                if cv2.waitKey(1) == ord('q'): exit()
                continue
            
            # --- BẮT ĐẦU TÍNH TOÁN (Chỉ chạy mỗi 5 frame 1 lần) ---
            
            # 1. Landmark Model
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark = pre_process_landmark(landmark_list)
            input_landmark = np.array([pre_processed_landmark], dtype=np.float32)
            scores_landmark = model_landmark.predict(input_landmark, verbose=0)[0]

            # 2. CNN Model
            x1, y1, x2, y2, size = get_bbox(landmark_list, h, w, padding=30)
            roi_square = np.zeros((size, size, 3), dtype=np.uint8)
            src_x1, src_y1 = max(0, x1), max(0, y1)
            src_x2, src_y2 = min(w, x2), min(h, y2)
            
            scores_cnn = np.zeros(len(CLASSES))
            
            if (src_x2 > src_x1) and (src_y2 > src_y1):
                crop = frame[src_y1:src_y2, src_x1:src_x2]
                dst_x1, dst_y1 = max(0, -x1), max(0, -y1)
                roi_square[dst_y1:dst_y1 + (src_y2-src_y1), dst_x1:dst_x1 + (src_x2-src_x1)] = crop
                
                img_input = cv2.resize(roi_square, (128, 128))
                img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                img_input = np.expand_dims(img_input, axis=0)
                
                # --- SỬA LỖI Ở ĐÂY: KHÔNG GỌI preprocess_input NỮA ---
                # Vì trong model.h5 đã có lớp đó rồi
                scores_cnn = model_cnn.predict(img_input, verbose=0)[0]

            # 3. Kết hợp
            final_scores = (scores_landmark * WEIGHT_LANDMARK) + (scores_cnn * WEIGHT_CNN)
            class_id = np.argmax(final_scores)
            current_confidence = final_scores[class_id]
            current_label = CLASSES[class_id]

            # Cập nhật màu sắc dựa trên kết quả mới
            if current_confidence > CONFIDENCE_THRESHOLD:
                current_color = (0, 255, 0) # Xanh
            else:
                current_label = "..."
                current_color = (0, 0, 255) # Đỏ

            # Vẽ kết quả mới
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), current_color, 2)
            cv2.putText(debug_image, f"{current_label} ({current_confidence*100:.0f}%)", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)

    cv2.imshow('Final System', debug_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()