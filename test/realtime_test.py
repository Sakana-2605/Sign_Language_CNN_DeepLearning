import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from PIL import ImageFont, ImageDraw, Image

# --- CẤU HÌNH TRỌNG SỐ ---
WEIGHT_STATIC = 0.7
WEIGHT_LSTM = 0.3

# --- CẤU HÌNH HỆ THỐNG ---
MODEL_STATIC_PATH = 'landmark_model.h5'
MODEL_LSTM_PATH = 'best_sign_model_cpu.keras'
CLASSES_STATIC_PATH = 'classes.npy'
FONT_PATH = "arial.ttf" # Đảm bảo file font này nằm cùng thư mục code

CONF_THRESH = 0.8  
DELAY_TIME = 1.2    
SEQUENCE_LENGTH = 30
STABILITY_FRAME_COUNT = 3 
OFFSET = 20 

# --- LOGIC TIẾNG VIỆT ---
VIETNAMESE_LOGIC = {
    ('U', 'MOC'): 'Ư', ('O', 'MOC'): 'Ơ', ('O', 'MU'): 'Ô',
    ('A', 'MU'): 'Â', ('E', 'MU'): 'Ê', ('A', 'TRANG'): 'Ă',
    ('Ư', 'NANG'): 'Ự', ('Ư', 'SAC'): 'Ứ', ('Ư', 'HUYEN'): 'Ừ', ('Ư', 'HOI'): 'Ử', ('Ư', 'NGA'): 'Ữ',
    ('Ơ', 'NANG'): 'Ợ', ('Ơ', 'SAC'): 'Ớ', ('Ơ', 'HUYEN'): 'Ờ', ('Ơ', 'HOI'): 'Ở', ('Ơ', 'NGA'): 'Ỡ',
    ('Ô', 'SAC'): 'Ố', ('Ô', 'HUYEN'): 'Ồ', ('Ô', 'NANG'): 'Ộ',
    ('Â', 'SAC'): 'Ấ', ('Â', 'HUYEN'): 'Ầ', ('Â', 'NANG'): 'Ậ',
    ('Ê', 'SAC'): 'Ế', ('Ê', 'HUYEN'): 'Ề', ('Ê', 'NANG'): 'Ệ',
    ('Ă', 'SAC'): 'Ắ', ('Ă', 'HUYEN'): 'Ằ',
    ('A', 'SAC'): 'Á', ('A', 'HUYEN'): 'À', ('A', 'NANG'): 'Ạ', 
    ('E', 'SAC'): 'É', ('U', 'SAC'): 'Ú', ('O', 'SAC'): 'Ó', ('I', 'SAC'): 'Í'
}

# --- CÁC HÀM HỖ TRỢ ---
def get_hand_bbox(hand_landmarks, h, w):
    """Tính toán khung bao quanh bàn tay"""
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    x1, y1 = int(min(x_coords)) - OFFSET, int(min(y_coords)) - OFFSET
    x2, y2 = int(max(x_coords)) + OFFSET, int(max(y_coords)) + OFFSET
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

def normalize_landmarks(landmarks):
    """Chuẩn hóa tọa độ landmarks về gốc 0 và scale"""
    landmarks = np.array(landmarks).reshape(-1, 3)
    landmarks = landmarks - landmarks[0]
    max_val = np.abs(landmarks).max()
    if max_val > 0: landmarks = landmarks / max_val
    return landmarks.flatten()

def draw_vietnamese_text(img, text, position, font_size, color):
    """Hỗ trợ vẽ chữ tiếng Việt bằng Pillow"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- KHỞI TẠO MODEL & MEDIAPIPE ---
try:
    model_static = tf.keras.models.load_model(MODEL_STATIC_PATH)
    model_lstm = tf.keras.models.load_model(MODEL_LSTM_PATH)
    actions_static = np.load(CLASSES_STATIC_PATH)
    actions_lstm = np.array(['SAC', 'HUYEN', 'HOI', 'NGA', 'NANG'])
    print("--- Đã tải các Model thành công ---")
except Exception as e:
    print(f"Lỗi khởi tạo: {e}")
    exit()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, 
    model_complexity=0, # Tối ưu tốc độ thực tế
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)

# --- BIẾN ĐIỀU KHIỂN ---
sentence = []
sequence_buffer = []      
prediction_history = []   
last_prediction = ""
last_time_recognized = 0
show_sentence = False # Mặc định ẩn văn bản

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    current_candidate = "None"
    current_conf = 0.0
    bbox = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            # 1. CHỈ XỬ LÝ TAY PHẢI
            if hand_handedness.classification[0].label == 'Right':
                hand_lms = results.multi_hand_landmarks[idx]
                bbox = get_hand_bbox(hand_lms, h, w)
                
                # Vẽ Skeleton
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                # Tiền xử lý
                raw_lms = [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
                processed = normalize_landmarks(raw_lms)
                sequence_buffer.append(processed)
                sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]
                
                # 2. LOGIC TRỌNG SỐ (WEIGHTED ENSEMBLE)
                res_static = model_static.predict(np.array([processed]), verbose=0)[0]
                
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    movement = np.std(sequence_buffer, axis=0).mean()
                    if movement > 0.06: # Có chuyển động: Ưu tiên LSTM
                        res_lstm = model_lstm.predict(np.expand_dims(sequence_buffer, axis=0), verbose=0)[0]
                        
                        idx_l = np.argmax(res_lstm)
                        idx_s = np.argmax(res_static)
                        
                        # So sánh điểm số có trọng số
                        if (res_lstm[idx_l] * WEIGHT_LSTM) > (res_static[idx_s] * WEIGHT_STATIC):
                            current_candidate = actions_lstm[idx_l]
                            current_conf = res_lstm[idx_l]
                        else:
                            current_candidate = actions_static[idx_s]
                            current_conf = res_static[idx_s]
                    else:
                        idx_s = np.argmax(res_static)
                        current_candidate = actions_static[idx_s]
                        current_conf = res_static[idx_s]
                else:
                    idx_s = np.argmax(res_static)
                    current_candidate = actions_static[idx_s]
                    current_conf = res_static[idx_s]

    # 3. KIỂM TRA ĐỘ ỔN ĐỊNH VÀ GỘP CHỮ
    if current_conf < CONF_THRESH: current_candidate = "None"
    
    prediction_history.append(current_candidate)
    prediction_history = prediction_history[-STABILITY_FRAME_COUNT:]
    
    if len(prediction_history) == STABILITY_FRAME_COUNT and len(set(prediction_history)) == 1:
        final_sign = prediction_history[0]
        if final_sign != "None" and (time.time() - last_time_recognized) > DELAY_TIME:
            if final_sign != last_prediction:
                if sentence and (sentence[-1], final_sign) in VIETNAMESE_LOGIC:
                    sentence[-1] = VIETNAMESE_LOGIC[(sentence[-1], final_sign)]
                elif final_sign == 'DD': sentence.append('Đ')
                elif final_sign not in ['MU', 'MOC', 'TRANG', 'SAC', 'HUYEN', 'HOI', 'NGA', 'NANG', 'None']:
                    sentence.append(final_sign)
                last_prediction, last_time_recognized = final_sign, time.time()

    # 4. HIỂN THỊ UI
    if bbox is not None and current_candidate != "None":
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{current_candidate} {current_conf:.2f}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    if show_sentence:
        res_txt = "Văn bản: " + "".join(sentence)
        frame = draw_vietnamese_text(frame, res_txt, (30, h - 70), 40, (0, 255, 255))
    
    # Hướng dẫn nhanh trên màn hình
    info = "T: Hien/An Van ban | C: Xoa | ESC: Thoat"
    cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('VSL - Right Hand System', frame)
    
    key = cv2.waitKey(1)
    if key == 27: break
    if key == ord('t'): show_sentence = not show_sentence
    if key == ord('c'): sentence = []

cap.release()
cv2.destroyAllWindows()