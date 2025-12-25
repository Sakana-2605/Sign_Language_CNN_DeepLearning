import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import copy
import itertools
import time
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- GIỮ NGUYÊN CHECK GPU GỐC ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Chỉ cấp phát VRAM khi cần thiết (tránh lỗi tràn bộ nhớ)
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ ĐÃ KÍCH HOẠT GPU: {len(gpus)} thiết bị found.")
        print(f"Chi tiết: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("❌ CẢNH BÁO: Không tìm thấy GPU, hệ thống sẽ chạy chậm trên CPU!")

# --- CẤU HÌNH ---
MODEL_CNN_PATH = 'models/sign_language_model.h5'
MODEL_LANDMARK_PATH = 'models/keypoint_classifier.h5'
CLASSES = ['A','B','C','D','DD','E','G',
           'H','I','K','L','M','N','O','P',
           'Q','R','S','T','U','V','X',
           'Y','MOC','MU','TRANG','None'] 

SKIP_FRAMES = 5 
WEIGHT_LANDMARK = 0.3
WEIGHT_CNN = 0.7
CONFIDENCE_THRESHOLD = 0.65

# --- LOAD MODELS ---
print("⏳ Đang load models...")
model_cnn = tf.keras.models.load_model(MODEL_CNN_PATH)
model_landmark = tf.keras.models.load_model(MODEL_LANDMARK_PATH)
print("✅ Sẵn sàng!")

mp_hands = mp.solutions.hands
# Cấu hình max_num_hands=1 để MediaPipe tập trung vào 1 tay duy nhất
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

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
    return list(map(normalize_, flatten_list))

def get_bbox(landmark_list, h, w, padding=30):
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

    # Kiểm tra cả landmarks và handedness
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            # MediaPipe phân loại 'Left'/'Right' dựa trên frame thực tế.
            # Vì ta đã flip(frame, 1), tay PHẢI thật sẽ được MediaPipe hiểu là "Left".
            hand_label = handedness.classification[0].label 
            
            if hand_label == "Left": # "Left" ở đây ứng với tay phải thật sau khi lật ảnh
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                x1, y1, x2, y2, size = get_bbox(landmark_list, h, w, padding=30)
                
                # Vẽ khung xương mỗi frame để duy trì độ mượt hiển thị
                mp.solutions.drawing_utils.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # --- LOGIC SKIP FRAME CHO PREDICTION ---
                frame_count += 1
                if frame_count % SKIP_FRAMES == 0:
                    # 1. Landmark Model
                    pre_processed_landmark = pre_process_landmark(landmark_list)
                    input_landmark = np.array([pre_processed_landmark], dtype=np.float32)
                    scores_landmark = model_landmark.predict(input_landmark, verbose=0)[0]

                    # 2. CNN Model
                    roi_square = np.zeros((size, size, 3), dtype=np.uint8)
                    src_x1, src_y1 = max(0, x1), max(0, y1)
                    src_x2, src_y2 = min(w, x2), min(h, y2)
                    
                    if (src_x2 > src_x1) and (src_y2 > src_y1):
                        crop = frame[src_y1:src_y2, src_x1:src_x2]
                        dst_x1, dst_y1 = max(0, -x1), max(0, -y1)
                        roi_square[dst_y1:dst_y1 + (src_y2-src_y1), dst_x1:dst_x1 + (src_x2-src_x1)] = crop
                        
                        img_input = cv2.resize(roi_square, (128, 128))
                        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                        img_input = np.expand_dims(img_input, axis=0)
                        scores_cnn = model_cnn.predict(img_input, verbose=0)[0]

                        # 3. Kết hợp kết quả
                        final_scores = (scores_landmark * WEIGHT_LANDMARK) + (scores_cnn * WEIGHT_CNN)
                        class_id = np.argmax(final_scores)
                        current_confidence = final_scores[class_id]

                        if current_confidence > CONFIDENCE_THRESHOLD:
                            current_label = CLASSES[class_id]
                            current_color = (0, 255, 0)
                        else:
                            current_label = "..."
                            current_color = (0, 0, 255)

                # Hiển thị BBox và kết quả (sử dụng giá trị current_ để giữ lại kết quả trong lúc skip)
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), current_color, 2)
                cv2.putText(debug_image, f"{current_label} ({current_confidence*100:.0f}%)", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)
            else:
                # Nếu đưa tay trái vào, chỉ vẽ thông báo mờ hoặc bỏ qua hoàn toàn
                cv2.putText(debug_image, "Only Right Hand", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Final System', debug_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()