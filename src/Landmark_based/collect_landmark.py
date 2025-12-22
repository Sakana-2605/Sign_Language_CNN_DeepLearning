import csv
import copy
import itertools
import cv2
import mediapipe as mp

# --- CẤU HÌNH ---
DATA_FILE = "data/keypoints.csv"
CLASSES = ['A', 'B', 'None'] # Định nghĩa các nhãn bạn muốn train

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Chuyển đổi sang toạ độ pixel
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 1. Chuyển về toạ độ tương đối (Relative Coordinates)
    # Lấy điểm cổ tay (điểm 0) làm gốc (0,0)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]

    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 2. Chuẩn hoá về khoảng [-1, 1] (Normalization)
    flatten_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, flatten_list)))

    def normalize_(n):
        return n / max_value

    flatten_list = list(map(normalize_, flatten_list))
    return flatten_list

def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    print("Nhấn các phím số để lưu dữ liệu:")
    for idx, name in enumerate(CLASSES):
        print(f"Phím {idx}: Lưu nhãn '{name}'")
    print("Phím 'q': Thoát")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ khớp tay
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Tính toán toạ độ chuẩn hoá
                landmark_list = calc_landmark_list(frame, hand_landmarks)
                pre_processed_landmark = pre_process_landmark(landmark_list)
                
                # Xử lý phím bấm để lưu
                key = cv2.waitKey(10)
                if 48 <= key <= 57: # ASCII từ phím '0' đến '9'
                    class_id = key - 48
                    if class_id < len(CLASSES):
                        # Lưu vào CSV
                        with open(DATA_FILE, 'a', newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([class_id, *pre_processed_landmark])
                        print(f"Đã lưu mẫu cho: {CLASSES[class_id]}")

        cv2.imshow('Dataset Collector (Landmarks)', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()