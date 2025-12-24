import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- 1. CẤU HÌNH ---
DATA_FILE = "data/diacritics_data.csv"
SEQUENCE_LENGTH = 30  # Phải khớp với lúc thu thập
NUM_FEATURES = 42    # 21 landmarks * 2 (x, y)

# --- 2. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ---
df = pd.read_csv(DATA_FILE, header=None)

# Tách nhãn (y) và đặc trưng (X)
y_raw = df.iloc[:, 0].values
X_raw = df.iloc[:, 1:].values

# Gom nhóm dữ liệu thành các chuỗi (Sequences)
# Ví dụ: từ 3000 dòng phẳng chuyển thành (100 mẫu, 30 frames, 42 features)
X = []
y = []

for i in range(0, len(X_raw), SEQUENCE_LENGTH):
    window = X_raw[i : i + SEQUENCE_LENGTH]
    if len(window) == SEQUENCE_LENGTH:
        X.append(window)
        y.append(y_raw[i]) # Nhãn của chuỗi là nhãn của frame đầu tiên

X = np.array(X)
y = to_categorical(y).astype(int)

# Chia dữ liệu Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. XÂY DỰNG MÔ HÌNH LSTM ---
model = Sequential([
    # Lớp LSTM đầu tiên (return_sequences=True để nối tiếp vào lớp LSTM sau)
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    Dropout(0.2),
    
    # Lớp LSTM thứ hai
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.2),
    
    # Lớp ẩn Dense
    Dense(64, activation='relu'),
    
    # Lớp đầu ra (Số lượng nhãn = y.shape[1])
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 4. HUẤN LUYỆN ---
print("Bắt đầu huấn luyện...")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# --- 5. LƯU MÔ HÌNH ---
model.save('sign_language_lstm.h5')
print("Đã lưu mô hình thành sign_language_lstm.h5")