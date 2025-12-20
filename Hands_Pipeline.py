import os
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

SRC_DIR = "data_aug"
DST_DIR = "data_pipeline"
os.makedirs(DST_DIR, exist_ok=True)

for cls in os.listdir(SRC_DIR):
    src_cls = os.path.join(SRC_DIR, cls)
    dst_cls = os.path.join(DST_DIR, cls)
    os.makedirs(dst_cls, exist_ok=True)

    for img_name in os.listdir(src_cls):
        img_path = os.path.join(src_cls, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            continue

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )

        cv2.imwrite(os.path.join(dst_cls, img_name), img)

print("Pipeline succeeded")
