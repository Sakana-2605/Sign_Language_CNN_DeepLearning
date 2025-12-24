import csv
import cv2
import mediapipe as mp
import copy
import itertools
import os
import time

# --- CẤU HÌNH LSTM ---
CLASSES = ['SAC', 'HUYEN', 'HOI', 'NGA', 'NANG']
SEQUENCE_LENGTH = 30  # Số khung hình cho mỗi hành động
SAMPLES_PER_CLASS = 20 # Số lần thực hiện mỗi dấu (mỗi lần là 1 chuỗi 30 frames)
DATA_FILE = "data/diacritics_data.csv"

if not os.path.exists('data'):
    os.makedirs('data')

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

def main():
    # --- TỰ ĐỘNG GÁN PHÍM SỐ ---
    key_map = {ord(str(i)): i for i in range(len(CLASSES))}
    
    print("="*45)
    for i, name in enumerate(CLASSES):
        print(f"Nhãn: {name:<10} | Nhấn phím số [{i}] để thu thập")
    print("="*45)

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        cv2.putText(display_frame, "READY - Press number key to start sequence", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('LSTM Data Collector', display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): break
        
        if key in key_map:
            class_id = key_map[key]
            print(f"\nChuẩn bị thu nhãn: {CLASSES[class_id]}")
            
            # Đếm ngược 1 giây để người dùng chuẩn bị tư thế tay
            for wait_time in range(3, 0, -1):
                ret, frame = cap.read()
                temp_f = cv2.flip(frame, 1)
                cv2.putText(temp_f, f"WAITING... {wait_time}", (200, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                cv2.imshow('LSTM Data Collector', temp_f)
                cv2.waitKey(300)

            # Bắt đầu thu 1 chuỗi (Sequence)
            sequence_buffer = []
            print(f"--- ĐANG QUAY ---")
            
            while len(sequence_buffer) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Tiền xử lý
                    image_width, image_height = frame.shape[1], frame.shape[0]
                    landmark_list = [[min(int(lm.x * image_width), image_width - 1),
                                      min(int(lm.y * image_height), image_height - 1)] 
                                     for lm in hand_landmarks.landmark]
                    
                    processed_data = pre_process_landmark(landmark_list)
                    sequence_buffer.append(processed_data)
                    
                    cv2.putText(frame, f"RECORDING: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "HAND NOT DETECTED!", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('LSTM Data Collector', frame)
                cv2.waitKey(1)

            # Lưu cả chuỗi vào CSV sau khi quay xong
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                with open(DATA_FILE, 'a', newline="") as f:
                    writer = csv.writer(f)
                    for frame_data in sequence_buffer:
                        writer.writerow([class_id, *frame_data])
                print(f"Đã lưu thành công 1 chuỗi cho {CLASSES[class_id]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()