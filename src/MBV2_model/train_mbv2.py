import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam

# --- CẤU HÌNH ---
DATA_DIR = './data_augmented' 
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = 'models/sign_language_model.h5'

os.makedirs('models', exist_ok=True)

def build_model(num_classes):
    # weights='imagenet': Sử dụng kiến thức đã học từ triệu ảnh thực tế
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Đóng băng các lớp dưới để giữ lại kiến thức cơ bản (feature extraction)
    base_model.trainable = False 
    
    # 2. Xây dựng phần đầu mới (Classification Head) cho bài toán của mình
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Preprocessing chuẩn của MobileNet (đưa pixel về khoảng -1 đến 1)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x) 
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x) 
    
    model = Model(inputs, outputs)
    return model

def main():
    print("Đang load dữ liệu...")
    
    train_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    val_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    class_names = train_ds.class_names
    print(f"Tìm thấy {len(class_names)} nhãn: {class_names}")
    
    with open('models/classes.txt', 'w') as f:
        f.write('\n'.join(class_names))

    # Tối ưu hiệu năng load dữ liệu
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # --- 2. XÂY DỰNG VÀ TRAIN MODEL ---
    model = build_model(len(class_names))
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nBắt đầu Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    
    # --- 3. LƯU MODEL ---
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()