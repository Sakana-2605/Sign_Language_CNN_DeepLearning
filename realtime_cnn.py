import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

MODEL_PATH = "model.h5"
IMG_SIZE = 128
CONF_THRESH = 0.75  
OFFSET = 20         

class_names = [
    'S','T','U','V','X','Y',
    'Â','Ă','Ê','Ô','Ơ','Ư',
    'Á','À','Ả','Ã','Ạ'
]

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("--- Model loaded successfully ---")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1, # 0: Nhanh, 1: Cân bằng (Khuyên dùng)
    min_detection_confidence=0.8, 
    min_tracking_confidence=0.5
)

def get_square_roi(frame, x1, y1, x2, y2):
    """Cắt ảnh thành hình vuông để tránh méo khi resize"""
    h_roi = y2 - y1
    w_roi = x2 - x1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    side = max(h_roi, w_roi)
    
    nx1 = max(0, cx - side // 2)
    ny1 = max(0, cy - side // 2)
    nx2 = min(frame.shape[1], nx1 + side)
    ny2 = min(frame.shape[0], ny1 + side)
    
    return frame[ny1:ny2, nx1:nx2], (nx1, ny1, nx2, ny2)

cap = cv2.VideoCapture(0)
p_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1) # Flip để soi gương cho tự nhiên
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # 1. Lấy tọa độ cực trị để vẽ ROI
            x_list, y_list = [], []
            for lm in hand_lms.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))
            
            x1, y1 = min(x_list) - OFFSET, min(y_list) - OFFSET
            x2, y2 = max(x_list) + OFFSET, max(y_list) + OFFSET

            # 2. Xử lý ROI vuông và Predict
            try:
                roi_raw, coords = get_square_roi(frame, x1, y1, x2, y2)
                rx1, ry1, rx2, ry2 = coords
                
                # Tiền xử lý chuyên sâu
                img_input = cv2.resize(roi_raw, (IMG_SIZE, IMG_SIZE))
                img_input = img_input.astype('float32') / 255.0
                img_input = np.expand_dims(img_input, axis=0)

                # Dự đoán
                prediction = model.predict(img_input, verbose=0)[0]
                idx = np.argmax(prediction)
                conf = prediction[idx]
                
                # 3. Hiển thị Skeleton TRỰC TIẾP lên Frame chính
                mp_draw.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

                # 4. Vẽ UI
                label = class_names[idx] if conf > CONF_THRESH else "Scanning..."
                color = (0, 255, 0) if conf > CONF_THRESH else (0, 165, 255)
                
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (rx1, ry1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            except Exception as e:
                pass

    # Tính FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()