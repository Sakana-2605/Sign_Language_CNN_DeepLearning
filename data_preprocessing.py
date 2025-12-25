import cv2
import mediapipe as mp
import os
import glob
from tqdm import tqdm

# --- CẤU HÌNH ---
INPUT_DIR = "data"               # Thư mục chứa ảnh gốc
OUTPUT_DIR = "cropped_mediapipe" # Thư mục lưu ảnh đã crop
PADDING = 10                     # Padding thêm vào (pixel)
CONFIDENCE = 0.5                 # Độ tin cậy tối thiểu (0.0 - 1.0)
EXTS = ['*.jpg', '*.jpeg', '*.png']
# ----------------

def get_bbox_from_landmarks(hand_landmarks, img_w, img_h, padding=10):
    """
    Tính toán Bounding Box từ các điểm landmarks (xương tay).
    MediaPipe trả về tọa độ 0.0 - 1.0, cần nhân với kích thước ảnh.
    """
    x_list = []
    y_list = []

    # Duyệt qua 21 điểm landmark của bàn tay
    for lm in hand_landmarks.landmark:
        x_list.append(lm.x)
        y_list.append(lm.y)

    # Tìm min/max
    x_min = min(x_list) * img_w
    x_max = max(x_list) * img_w
    y_min = min(y_list) * img_h
    y_max = max(y_list) * img_h

    # Chuyển sang số nguyên và thêm Padding
    x_min = int(max(0, x_min - padding))
    y_min = int(max(0, y_min - padding))
    x_max = int(min(img_w, x_max + padding))
    y_max = int(min(img_h, y_max + padding))

    return x_min, y_min, x_max, y_max

def main():
    # 1. Khởi tạo MediaPipe Hands
    mp_hands = mp.solutions.hands
    # static_image_mode=True: Tối ưu cho ảnh tĩnh (chính xác hơn video mode)
    # max_num_hands=2: Số lượng tay tối đa muốn detect trong 1 ảnh
    hands = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=4, 
        min_detection_confidence=CONFIDENCE
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Quét file ảnh
    image_files = []
    for ext in EXTS:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, '**', ext), recursive=True))

    print(f"Tìm thấy {len(image_files)} ảnh. Đang xử lý bằng MediaPipe...")
    
    count_crops = 0

    for img_path in tqdm(image_files):
        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w, _ = img.shape

        # MediaPipe yêu cầu ảnh RGB (OpenCV mặc định là BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Xử lý Detect
        results = hands.process(img_rgb)

        # Nếu phát hiện có tay
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                # 4. Tính toán tọa độ cắt
                x1, y1, x2, y2 = get_bbox_from_landmarks(hand_landmarks, w, h, PADDING)

                # Kiểm tra kích thước hợp lệ
                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2]

                    # 5. Lưu ảnh vào thư mục con giống cấu trúc trong `data/`
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    # Lấy thư mục con đầu tiên dưới INPUT_DIR (ví dụ: data/A/..., lấy 'A')
                    rel_path = os.path.relpath(img_path, INPUT_DIR)
                    parts = rel_path.split(os.sep)
                    if len(parts) > 1:
                        class_dir = parts[0]
                    else:
                        class_dir = "uncategorized"

                    dest_dir = os.path.join(OUTPUT_DIR, class_dir)
                    os.makedirs(dest_dir, exist_ok=True)

                    save_name = f"{base_name}_mp_{i}.jpg"
                    save_path = os.path.join(dest_dir, save_name)

                    cv2.imwrite(save_path, crop)
                    count_crops += 1

    hands.close()
    print(f"\nHoàn tất! Đã crop được {count_crops} bàn tay.")
    print(f"Kiểm tra tại thư mục: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()