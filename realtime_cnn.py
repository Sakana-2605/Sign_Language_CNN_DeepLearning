# ======================
# FIX protobuf issue
# ======================
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# ======================
# CONFIG
# ======================
MODEL_PATH = "model.h5"
IMG_SIZE = 128
CONF_THRESH = 0.6

class_names = [
    'S','T','U','V','X','Y',
    'Â','Ă','Ê','Ô','Ơ','Ư',
    'Á','À','Ả','Ã','Ạ'
]

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(MODEL_PATH)

# ======================
# MEDIAPIPE
# ======================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ======================
# PREPROCESS
# ======================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ======================
# WEBCAM
# ======================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        xs = [int(lm.x * w) for lm in hand.landmark]
        ys = [int(lm.y * h) for lm in hand.landmark]

        pad = 30
        x1, x2 = max(0, min(xs)-pad), min(w, max(xs)+pad)
        y1, y2 = max(0, min(ys)-pad), min(h, max(ys)+pad)

        roi = frame[y1:y2, x1:x2].copy()

        # overlay skeleton
        mp_draw.draw_landmarks(
            roi,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style()
        )

        pred = model.predict(preprocess(roi), verbose=0)[0]
        idx = np.argmax(pred)
        conf = float(pred[idx])
        label = class_names[idx] if conf > CONF_THRESH else "?"

        frame[y1:y2, x1:x2] = roi
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(
            frame,
            f"{label} ({conf:.2f})",
            (x1, y1-10 if y1 > 30 else y1+30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,0,255),
            2
        )

    cv2.imshow("Sign Language CNN - Realtime", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()