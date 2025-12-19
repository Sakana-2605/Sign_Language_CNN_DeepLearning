import os
import cv2
import time

DATA_DIR = './data'
DATASET_SIZE = 100
CAMERA_INDEX = 0

COUNTDOWN_TIME = 3      # giây để chuẩn bị
CAPTURE_DELAY = 200    # ms giữa mỗi ảnh

ROI_SIZE = 300          # ROI bàn tay

SIGNS = [
    'A','B','C','D','Đ','E','G','H',
    'I','K','L','M','N','O','P','Q',
    'R','S','T','U','V','X','Y',
    'Â','Ê','Ô','Ơ','Ư','Ă'
]

os.makedirs(DATA_DIR, exist_ok=True)
for sign in SIGNS:
    os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

for sign in SIGNS:
    print(f'\n📸 Collecting data for sign: {sign}')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        x1 = cx - ROI_SIZE // 2
        y1 = cy - ROI_SIZE // 2
        x2 = cx + ROI_SIZE // 2
        y2 = cy + ROI_SIZE // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Ready for {sign} - Press Q',
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 0), 3)

        cv2.imshow('Capture', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    for t in range(COUNTDOWN_TIME, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Start in {t}',
                    (w//2 - 120, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 4)

        cv2.imshow('Capture', frame)
        cv2.waitKey(1000)

    for i in range(1, DATASET_SIZE + 1):
        ret, frame = cap.read()
        if not ret:
            continue

        roi = frame[y1:y2, x1:x2]

        filename = f'{sign}_{i:03d}.jpg'
        filepath = os.path.join(DATA_DIR, sign, filename)
        cv2.imwrite(filepath, roi)

        cv2.putText(roi, filename,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

        cv2.imshow('ROI', roi)
        cv2.waitKey(CAPTURE_DELAY)

cap.release()
cv2.destroyAllWindows()
