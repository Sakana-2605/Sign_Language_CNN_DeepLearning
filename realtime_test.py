import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import copy
import itertools
import time
import os
from PIL import ImageFont, ImageDraw, Image

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- CẤU HÌNH HỆ THỐNG ---
MODEL_CNN_PATH = 'models/sign_language_model.h5'
MODEL_LANDMARK_PATH = 'models/keypoint_classifier.h5'
# Danh sách nhãn (Phải khớp với file classes.npy khi train)
CLASSES = ['A', 'B', 'U', 'O', 'MU', 'MOC', 'SAC', 'HUYEN', 'NANG', 'HOI', 'NGA', 'TRANG', 'None'] 

# BẢNG TRA CỨU TELEX NÂNG CAO (Xử lý dấu chồng dấu)
VIETNAMESE_LOGIC = {
    # Tầng 1: Chữ cái đơn + Chữ cái có dấu mũ
    ('U', 'MOC'): 'Ư', ('O', 'MOC'): 'Ơ', ('O', 'MU'): 'Ô',
    ('A', 'MU'): 'Â', ('E', 'MU'): 'Ê', ('A', 'TRANG'): 'Ă', ('D', 'GACH'): 'Đ',
    
    # Tầng 2: Chữ cái đôi có dấu mũ + Dấu thanh
    ('Ư', 'NANG'): 'Ự', ('Ư', 'SAC'): 'Ứ', ('Ư', 'HUYEN'): 'Ừ', ('Ư', 'HOI'): 'Ử', ('Ư', 'NGA'): 'Ữ',
    ('Ơ', 'NANG'): 'Ợ', ('Ơ', 'SAC'): 'Ớ', ('Ơ', 'HUYEN'): 'Ờ', ('Ơ', 'HOI'): 'Ở', ('Ơ', 'NGA'): 'Ỡ',
    ('Ô', 'SAC'): 'Ố', ('Ô', 'HUYEN'): 'Ồ', ('Â', 'SAC'): 'Ấ', ('Ê', 'SAC'): 'Ế', ('Ă', 'SAC'): 'Ắ',
    
    # Tầng 3: Chữ cái đơn + Dấu thanh
    ('A', 'SAC'): 'Á', ('A', 'HUYEN'): 'À', ('A', 'NANG'): 'Ạ', ('E', 'SAC'): 'É', ('U', 'SAC'): 'Ú'
}

# Cấu hình AI
SKIP_FRAMES = 5 
WEIGHT_LANDMARK = 0.7
WEIGHT_CNN = 0.3
CONFIDENCE_THRESHOLD = 0.7
DELAY_TIME = 2.0  # Giây nghỉ để kịp đổi tư thế tay
FONT_PATH = "arial.ttf" # Đường dẫn font tiếng Việt

# --- LOAD MODELS ---
print("⏳ Đang load models...")
model_cnn = tf.keras.models.load_model(MODEL_CNN_PATH)
model_landmark = tf.keras.models.load_model(MODEL_LANDMARK_PATH)
print("✅ Sẵn sàng!")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- HÀM VẼ CHỮ TIẾNG VIỆT SỬ DỤNG PILLOW ---
def draw_vietnamese_text(img, text, position, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- CÁC HÀM TIỀN XỬ LÝ ---
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

def get_bbox(landmark_list, h, w, padding=30):
    pts = np.array(landmark_list)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    box_w, box_h = x_max - x_min, y_max - y_min
    max_side = max(box_w, box_h) + padding * 2
    center_x, center_y = x_min + box_w // 2, y_min + box_h // 2
    x1, y1 = center_x - max_side // 2, center_y - max_side // 2
    return int(x1), int(y1), int(x1 + max_side), int(y1 + max_side), int(max_side)

# --- VÒNG LẶP CHÍNH ---
cap = cv2.VideoCapture(0)
frame_count = 0
sentence = []
last_prediction = ""
last_time_recognized = 0

current_label = "..."
current_confidence = 0.0
current_color = (0, 0, 255)

SPECIAL_SIGNS = ['MU', 'MOC', 'SAC', 'HUYEN', 'NANG', 'HOI', 'NGA', 'TRANG', 'GACH']

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    debug_image = copy.deepcopy(frame)
    h, w, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            frame_count += 1
            
            # CHỈ TÍNH TOÁN AI MỖI 5 FRAME
            if frame_count % SKIP_FRAMES == 0:
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                # 1. Dự đoán Landmark
                pre_processed = pre_process_landmark(landmark_list)
                scores_landmark = model_landmark.predict(np.array([pre_processed]), verbose=0)[0]

                # 2. Dự đoán CNN
                x1, y1, x2, y2, size = get_bbox(landmark_list, h, w)
                roi_square = np.zeros((size, size, 3), dtype=np.uint8)
                sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                if (sx2 > sx1) and (sy2 > sy1):
                    crop = frame[sy1:sy2, sx1:sx2]
                    dst_x1, dst_y1 = max(0, -x1), max(0, -y1)
                    roi_square[dst_y1:dst_y1 + (sy2-sy1), dst_x1:dst_x1 + (sx2-sx1)] = crop
                    img_input = np.expand_dims(cv2.resize(cv2.cvtColor(roi_square, cv2.COLOR_BGR2RGB), (128, 128)), axis=0)
                    scores_cnn = model_cnn.predict(img_input, verbose=0)[0]
                else:
                    scores_cnn = np.zeros(len(CLASSES))

                # 3. Kết hợp Ensemble
                final_scores = (scores_landmark * WEIGHT_LANDMARK) + (scores_cnn * WEIGHT_CNN)
                idx = np.argmax(final_scores)
                current_confidence = final_scores[idx]
                current_label = CLASSES[idx]

                # 4. LOGIC GHÉP CHỮ VÀ DELAY
                if current_confidence > CONFIDENCE_THRESHOLD:
                    now = time.time()
                    if (now - last_time_recognized) > DELAY_TIME:
                        if current_label != last_prediction and current_label != "None":
                            
                            if sentence:
                                pair = (sentence[-1], current_label)
                                if pair in VIETNAMESE_LOGIC:
                                    sentence[-1] = VIETNAMESE_LOGIC[pair]
                                elif current_label not in SPECIAL_SIGNS:
                                    sentence.append(current_label)
                            else:
                                if current_label not in SPECIAL_SIGNS:
                                    sentence.append(current_label)
                            
                            last_prediction = current_label
                            last_time_recognized = now
                    current_color = (0, 255, 0)
                else:
                    current_color = (0, 0, 255)

            # Vẽ kết quả tạm thời lên khung tay
            l_list = calc_landmark_list(debug_image, hand_landmarks)
            bx1, by1, bx2, by2, _ = get_bbox(l_list, h, w)
            cv2.rectangle(debug_image, (bx1, by1), (bx2, by2), current_color, 2)
            cv2.putText(debug_image, f"{current_label} ({current_confidence*100:.0f}%)", 
                        (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)

    # HIỂN THỊ CÂU TIẾNG VIỆT CHUẨN
    output_str = "".join(sentence)
    debug_image = draw_vietnamese_text(debug_image, f"Result: {output_str}", (20, h - 70), 45, (0, 255, 255))

    cv2.imshow('Vietnamese Gesture System', debug_image)
    key = cv2.waitKey(1)
    if key == ord('q'): break
    if key == ord('c'): sentence = [] # Xóa câu

cap.release()
cv2.destroyAllWindows()