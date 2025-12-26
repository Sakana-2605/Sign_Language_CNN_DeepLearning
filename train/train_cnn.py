# import tensorflow as tf
# from pathlib import Path

# IMG_SIZE = 128
# BATCH_SIZE = 32
# EPOCHS_1 = 15   # train head
# EPOCHS_2 = 10   # fine-tune
# input_dir = "data_cnn"


# def build_image_dataset_from_directory(directory, image_size=(128, 128), batch_size=32, shuffle=True):
#     p = Path(directory)
#     if not p.exists():
#         raise ValueError(f"Directory not found: {directory}")

#     # collect class directories in deterministic order
#     class_dirs = [d for d in sorted(p.iterdir()) if d.is_dir()]
#     class_names = [d.name for d in class_dirs]

#     file_paths = []
#     labels = []
#     exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
#     for idx, d in enumerate(class_dirs):
#         for ext in exts:
#             for f in d.glob(ext):
#                 file_paths.append(str(f))
#                 labels.append(idx)

#     if len(file_paths) == 0:
#         raise ValueError(f"No image files found under {directory}")

#     ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(file_paths))

#     def _process(path, label):
#         image = tf.io.read_file(path)
#         image = tf.image.decode_image(image, channels=3)
#         image.set_shape([None, None, 3])
#         image = tf.image.resize(image, image_size)
#         return image, label

#     ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
#     ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     # attach class_names attribute so existing code can read it
#     ds.class_names = class_names
#     return ds


# train_ds = build_image_dataset_from_directory(f"{input_dir}/train", image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

# val_ds = build_image_dataset_from_directory(f"{input_dir}/val", image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False)

# NUM_CLASSES = len(train_ds.class_names)

# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     include_top=False,
#     weights="imagenet"
# )

# base_model.trainable = False

# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(1./255),
#     base_model,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
# ])

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-3),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# print("🚀 Training head...")
# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS_1
# )

# print("🔧 Fine-tuning...")
# base_model.trainable = True
# for layer in base_model.layers[:-30]:
#     layer.trainable = False

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-4),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS_2
# )

# model.save("model.h5")
# print("Model saved as model.h5")


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf

# # 1. Hàm chuẩn hóa tọa độ (CỰC KỲ QUAN TRỌNG)
# def normalize_landmarks(landmarks):
#     # landmarks: mảng 63 phần tử (x,y,z * 21)
#     landmarks = np.array(landmarks).reshape(-1, 3)
#     # Gốc tọa độ là cổ tay
#     base_node = landmarks[0]
#     landmarks = landmarks - base_node
#     # Max distance để scale về [-1, 1]
#     max_value = np.abs(landmarks).max()
#     if max_value > 0:
#         landmarks = landmarks / max_value
#     return landmarks.flatten()

# # 2. Load và tiền xử lý
# df = pd.read_csv('hand_landmarks.csv')
# X_raw = df.iloc[:, :-1].values
# y_raw = df.iloc[:, -1].values

# X = np.array([normalize_landmarks(x) for x in X_raw])
# encoder = LabelEncoder()
# y = encoder.fit_transform(y_raw)

# # 3. Augmentation cho dữ liệu số (Nhân bản data lên 5 lần)
# X_aug = []
# y_aug = []
# for i in range(len(X)):
#     for _ in range(5):
#         noise = np.random.normal(0, 0.005, X[i].shape) # Thêm nhiễu cực nhỏ
#         X_aug.append(X[i] + noise)
#         y_aug.append(y[i])

# X_aug = np.array(X_aug)
# y_aug = np.array(y_aug)

# X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, stratify=y_aug)

# # 4. Model MLP cải tiến
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(63,)),
#     tf.keras.layers.Dense(128, activation='leaky_relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(64, activation='leaky_relu'),
#     tf.keras.layers.Dense(32, activation='leaky_relu'),
#     tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
# ])

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss='sparse_categorical_crossentropy', 
#               metrics=['accuracy'])

# # Train nhiều epoch hơn vì dữ liệu tọa độ rất nhẹ
# model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test))
# model.save("hand_landmark_model.h5")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load data
X = np.load('X_data.npy')
y_raw = np.load('y_labels.npy')

# 2. Tiền xử lý nhãn
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
np.save('classes.npy', encoder.classes_) # Lưu tên các chữ cái

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 3. Khởi tạo Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Train
model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))
model.save('landmark_model.h5')