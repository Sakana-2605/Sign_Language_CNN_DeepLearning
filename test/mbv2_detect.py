import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os

# --- 1. CẤU HÌNH GPU (BẮT BUỘC) ---
# Fix lỗi tràn bộ nhớ khi load model
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ ĐÃ KÍCH HOẠT GPU: {len(gpus)} thiết bị.")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ Đang chạy trên CPU (Sẽ chậm hơn)")

# --- 2. CẤU HÌNH PARAMETERS ---
MODEL_PATH = 'models/sign_language_model.h5'
CLASS_PATH = 'models/classes.txt'
IMG_SIZE = (128, 128) # Phải khớp với lúc train
CONFIDENCE_THRESHOLD = 0.7 # Chỉ hiện kết quả nếu độ tin cậy > 70%

# --- 3. LOAD MODEL VÀ NHÃN ---
print("⏳ Đang load model & labels...")

# Load tên nhãn từ file classes.txt
try:
    with open(CLASS_PATH, 'r', encoding='utf-8') as f:
        class_names = f.read().splitlines()
    print(f"✅ Đã load {len(class_names)} nhãn: {class_names}")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file {CLASS_PATH}. Hãy chắc chắn bạn đã chạy file train!")
    exit()

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model đã sẵn sàng!")

# --- 4. KHỞI TẠO MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, # Chỉ nhận diện 1 tay để tránh loạn
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- 5. HÀM HỖ TRỢ ---
def get_bbox(landmarks, h, w, padding=20):
    """Tính toán vùng hình vuông bao quanh bàn tay"""
    x_list = [int(lm.x * w) for lm in landmarks.landmark]
    y_list = [int(lm.y * h) for lm in landmarks.landmark]

    x_min, x_max = min(x_list), max(x_list)
    y_min, y_max = min(y_list), max(y_list)

    box_w = x_max - x_min
    box_h = y_max - y_min
    
    # Làm cho khung thành hình vuông (để khi resize ảnh không bị méo)
    max_side = max(box_w, box_h) + padding * 2
    
    center_x = x_min + box_w // 2
    center_y = y_min + box_h // 2
    
    x1 = center_x - max_side // 2
    y1 = center_y - max_side // 2
    x2 = x1 + max_side
    y2 = y1 + max_side

    return x1, y1, x2, y2

# --- 6. MAIN LOOP ---
cap = cv2.VideoCapture(0)

# Biến tính FPS
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Lật ảnh cho giống gương
    frame = cv2.flip(frame, 1)
    debug_image = frame.copy()
    h, w, _ = frame.shape

    # Chuyển sang RGB để MediaPipe xử lý
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    label_text = "..."
    conf_score = 0.0
    color = (0, 0, 255) # Đỏ mặc định

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Vẽ khung xương khớp tay
            mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 2. Lấy tọa độ khung bao (Bounding Box)
            x1, y1, x2, y2 = get_bbox(hand_landmarks, h, w, padding=30)

            # 3. Kiểm tra biên (tránh lỗi crash khi tay ra khỏi màn hình)
            if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                # Cắt ảnh tay
                hand_crop = frame[y1:y2, x1:x2]
                
                if hand_crop.size > 0:
                    # 4. Pre-processing (Chuẩn bị ảnh cho model)
                    img_input = cv2.resize(hand_crop, IMG_SIZE)
                    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB) # Model train bằng RGB
                    img_input = np.expand_dims(img_input, axis=0) # Thêm chiều batch: (1, 128, 128, 3)

                    # 5. Predict (Dự đoán)
                    prediction = model.predict(img_input, verbose=0)[0]
                    class_id = np.argmax(prediction)
                    conf_score = prediction[class_id]
                    
                    if conf_score > CONFIDENCE_THRESHOLD:
                        label_text = class_names[class_id]
                        color = (0, 255, 0) # Xanh lá nếu tự tin

                # 6. Vẽ khung chữ nhật và kết quả
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ nền đen cho chữ dễ đọc
                cv2.rectangle(debug_image, (x1, y1 - 30), (x1 + 200, y1), color, -1)
                cv2.putText(debug_image, f"{label_text} ({conf_score*100:.1f}%)", 
                            (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Tính và hiện FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
    cv2.putText(debug_image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Sign Language Detection (MobileNetV2)", debug_image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()