import cv2
import mediapipe as mp
import numpy as np
import os
import time

DATA_PATH = 'MP_Data' 

actions = np.array(['SAC', 'HUYEN', 'HOI', 'NGA', 'NANG'])
no_sequences = 20  # Số lượng video cho mỗi dấu
sequence_length = 30 # Mỗi video gồm 30 khung hình

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Tạo thư mục lưu trữ
for action in actions: 
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

cap = cv2.VideoCapture(0)
with hands as hand_model:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hand_model.process(rgb)

                if frame_num == 0: 
                    cv2.putText(frame, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Collecting for {action} Video {sequence}', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    cv2.waitKey(1000) # Nghỉ 1s để bạn chuẩn bị tư thế
                else: 
                    cv2.putText(frame, f'Collecting for {action} Video {sequence}', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)

                # Trích xuất Landmark
                keypoints = np.zeros(63) 
                if results.multi_hand_landmarks:
                    # Lấy tay đầu tiên, biến thành mảng phẳng 63 phần tử (21 điểm * 3 tọa độ)
                    lms = results.multi_hand_landmarks[0].landmark
                    keypoints = np.array([[res.x, res.y, res.z] for res in lms]).flatten()

                # Lưu vào file .npy
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()