import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

# --- CẤU HÌNH ---
DATA_FILE = "data/keypoints.csv"
MODEL_SAVE_PATH = 'models/keypoint_classifier.h5'
CLASSES = ['A','B','C','D','DD','E','G',
           'H','I','K','L','M','N','O','P',
           'Q','R','S','T','U','V','X',
           'Y','MOC','MU','TRANG','None'] 

def main():
    # 1. Đọc dữ liệu
    # Dữ liệu không có header, cột đầu là label, 42 cột sau là toạ độ
    dataset = pd.read_csv(DATA_FILE, header=None)
    
    X = dataset.iloc[:, 1:].values # Features (Toạ độ)
    y = dataset.iloc[:, 0].values  # Labels

    # 2. Chia tập Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Xây dựng Model
    # Input: 42 điểm (21 toạ độ x, y)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((42, )),
        tf.keras.layers.Dense(20, activation='relu'), # Layer ẩn 1
        tf.keras.layers.Dropout(0.2),                 # Chống học vẹt
        tf.keras.layers.Dense(10, activation='relu'), # Layer ẩn 2
        tf.keras.layers.Dense(len(CLASSES), activation='softmax') # Output
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Training
    model.fit(
        X_train, y_train,
        epochs=25,             # Train 50 vòng (rất nhanh)
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # 5. Lưu model
    model.save(MODEL_SAVE_PATH)
    print("Training xong!")

if __name__ == "__main__":
    main()