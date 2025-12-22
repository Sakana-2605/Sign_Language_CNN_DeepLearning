import os
import cv2
import time
import mediapipe as mp
import numpy as np

# --- CẤU HÌNH ---
DATA_DIR = './data'
DATASET_SIZE = 250
CAMERA_INDEX = 0

COUNTDOWN_TIME = 3
CAPTURE_DELAY = 0.05

PADDING = 40

SIGNS = [
    "B"
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

os.makedirs(DATA_DIR, exist_ok=True)
for sign in SIGNS:
    os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

def get_bbox(landmarks, h, w, padding=20):
    x_list = [int(lm.x * w) for lm in landmarks]
    y_list = [int(lm.y * h) for lm in landmarks]

    min_x, max_x = min(x_list), max(x_list)
    min_y, max_y = min(y_list), max(y_list)

    # Chiều rộng và chiều cao thực tế của tay
    box_w = max_x - min_x
    box_h = max_y - min_y

    # Lấy cạnh lớn nhất để tạo hình vuông (giúp ảnh không bị méo khi resize sau này)
    max_side = max(box_w, box_h) + padding * 2

    # Tìm tâm của bàn tay
    center_x = min_x + box_w // 2
    center_y = min_y + box_h // 2

    # Tính toán toạ độ bắt đầu và kết thúc của khung vuông
    x1 = center_x - max_side // 2
    y1 = center_y - max_side // 2
    x2 = x1 + max_side
    y2 = y1 + max_side

    return x1, y1, x2, y2, max_side

for sign in SIGNS:
    print(f'\nĐang thu thập dữ liệu cho: {sign}')
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_status = "No Hand"
        color = (0, 0, 255) # Đỏ

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0].landmark
            
            # Gọi hàm tính toán khung bao
            x1, y1, x2, y2, size = get_bbox(hand_landmarks, h, w, PADDING)
            
            # Vẽ khung preview
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            hand_status = "Ready"
            color = (0, 255, 0) # Xanh

        cv2.putText(frame, f'Sign: {sign} | {hand_status}', (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, 'Nhan Q de bat dau', (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for t in range(COUNTDOWN_TIME, 0, -1):
        start_time = time.time()
        while time.time() - start_time < 1:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f'Get Ready: {t}', (w//2 - 120, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            cv2.imshow('Capture', frame)
            cv2.waitKey(1)

    i = 1
    while i <= DATASET_SIZE:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0].landmark
            x1, y1, x2, y2, size = get_bbox(hand_landmarks, h, w, PADDING)
            
            # Tạo ảnh nền đen kích thước vuông (size x size)
            roi_square = np.zeros((size, size, 3), dtype=np.uint8)
            
            # Tính toán phần giao nhau giữa khung vuông và frame webcam
            # Toạ độ trên frame
            src_x1 = max(0, x1)
            src_y1 = max(0, y1)
            src_x2 = min(w, x2)
            src_y2 = min(h, y2)

            # Kích thước phần giao nhau
            w_crop = src_x2 - src_x1
            h_crop = src_y2 - src_y1

            if w_crop > 0 and h_crop > 0:
                # Cắt ảnh từ frame
                crop = frame[src_y1:src_y2, src_x1:src_x2]
                
                # Tính toán vị trí dán vào ảnh vuông nền đen (để căn giữa)
                # Toạ độ trên roi_square
                dst_x1 = max(0, -x1) # Nếu x1 âm, thì lùi vào trong roi_square
                dst_y1 = max(0, -y1)
                
                # Dán ảnh crop vào roi_square
                roi_square[dst_y1:dst_y1+h_crop, dst_x1:dst_x1+w_crop] = crop

                # Lưu ảnh
                # Resize về kích thước chuẩn (VD: 224x224) để tiết kiệm ổ cứng nếu muốn
                # roi_final = cv2.resize(roi_square, (224, 224)) 
                
                filename = f'{sign}_{i:03d}.jpg'
                filepath = os.path.join(DATA_DIR, sign, filename)
                cv2.imwrite(filepath, roi_square) # Lưu ảnh vuông

                cv2.putText(frame, f'Saved: {i}/{DATASET_SIZE}', (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                i += 1
            else:
                 cv2.putText(frame, 'Hand out of bounds!', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            cv2.putText(frame, 'No Hand Detected', (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Capture', frame)
        # Delay nhẹ để không chụp quá nhanh (trùng frame)
        time.sleep(0.01) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()