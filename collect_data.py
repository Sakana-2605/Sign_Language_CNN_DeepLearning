# import os
# import cv2
# import time

# DATA_DIR = './data'
# DATASET_SIZE = 250
# CAMERA_INDEX = 0

# COUNTDOWN_TIME = 3      
# CAPTURE_DELAY = 200    

# ROI_SIZE = 200       

# SIGNS = [
#     'Â','Ă','Ê','Ô','Ơ','Ư',
#     'Á','À','Ả','Ã','Ạ'
# ]

# os.makedirs(DATA_DIR, exist_ok=True)
# for sign in SIGNS:
#     os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

# cap = cv2.VideoCapture(CAMERA_INDEX)

# for sign in SIGNS:
#     print(f'\n📸 Collecting data for sign: {sign}')

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         h, w, _ = frame.shape
#         cx, cy = w // 2, h // 2
#         x1 = cx - ROI_SIZE // 2
#         y1 = cy - ROI_SIZE // 2
#         x2 = cx + ROI_SIZE // 2
#         y2 = cy + ROI_SIZE // 2

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f'Ready for {sign} - Press Q',
#                     (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                     1.1, (0, 255, 0), 3)

#         cv2.imshow('Capture', frame)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

#     for t in range(COUNTDOWN_TIME, 0, -1):
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f'Start in {t}',
#                     (w//2 - 120, h//2),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     2, (0, 0, 255), 4)

#         cv2.imshow('Capture', frame)
#         cv2.waitKey(1000)

#     for i in range(1, DATASET_SIZE + 1):
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         roi = frame[y1:y2, x1:x2]

#         filename = f'{sign}_{i:03d}.jpg'
#         filepath = os.path.join(DATA_DIR, sign, filename)
#         cv2.imwrite(filepath, roi)

#         cv2.putText(roi, filename,
#                     (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.9, (255, 0, 0), 2)

#         cv2.imshow('ROI', roi)
#         cv2.waitKey(CAPTURE_DELAY)

# cap.release()
# cv2.destroyAllWindows()
import os
import cv2
import time
import mediapipe as mp

DATA_DIR = './data'
DATASET_SIZE = 250
CAMERA_INDEX = 0

COUNTDOWN_TIME = 3      
CAPTURE_DELAY = 200    

ROI_SIZE = 200 
MARGIN = 30 

SIGNS = [
    'AA','AW','EE','OO','OW','UW'
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

for sign in SIGNS:
    print(f'\n📸 Collecting data for sign: {sign}')

    while True:
        ret, frame = cap.read()
        if not ret: continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        x1, y1, x2, y2 = w//2-100, h//2-100, w//2+100, h//2+100

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            x_coords = [int(p.x * w) for p in lm]
            y_coords = [int(p.y * h) for p in lm]
            
            center_x, center_y = sum(x_coords)//21, sum(y_coords)//21
            x1, y1 = center_x - ROI_SIZE//2, center_y - ROI_SIZE//2
            x2, y2 = x1 + ROI_SIZE, y1 + ROI_SIZE

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Ready for {sign} - Press Q', (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

        cv2.imshow('Capture', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    for t in range(COUNTDOWN_TIME, 0, -1):
        ret, frame = cap.read()
        if not ret: continue
        cv2.putText(frame, f'Start in {t}', (w//2 - 120, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow('Capture', frame)
        cv2.waitKey(1000)

    for i in range(1, DATASET_SIZE + 1):
        ret, frame = cap.read()
        if not ret: continue
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            x_coords = [int(p.x * w) for p in lm]
            y_coords = [int(p.y * h) for p in lm]
            center_x, center_y = sum(x_coords)//21, sum(y_coords)//21
            x1, y1 = center_x - ROI_SIZE//2, center_y - ROI_SIZE//2
            x2, y2 = x1 + ROI_SIZE, y1 + ROI_SIZE

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]
        
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            roi = cv2.resize(roi, (ROI_SIZE, ROI_SIZE))

            filename = f'{sign}_{i:03d}.jpg'
            filepath = os.path.join(DATA_DIR, sign, filename)
            cv2.imwrite(filepath, roi)

            cv2.putText(roi, filename, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.imshow('ROI', roi)
        
        cv2.imshow('Capture', frame)
        cv2.waitKey(CAPTURE_DELAY)

cap.release()
cv2.destroyAllWindows()