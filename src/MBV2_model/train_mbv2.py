import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam

# --- 1. CẤU HÌNH GPU (QUAN TRỌNG) ---
# Đoạn này giúp tránh lỗi tràn bộ nhớ (OOM) trên Windows
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ ĐÃ KÍCH HOẠT GPU: {len(gpus)} thiết bị.")
        print(f"Chi tiết: {gpus}")
    except RuntimeError as e:
        print(f"Lỗi cấu hình GPU: {e}")
else:
    print("⚠️ CẢNH BÁO: Không tìm thấy GPU, quá trình train sẽ rất chậm trên CPU!")

# --- CẤU HÌNH PARAMETERS ---
DATA_DIR = './data_augmented' 
IMG_SIZE = (128, 128)
BATCH_SIZE = 32      # Nếu GPU yếu (ít VRAM), hãy giảm xuống 16
EPOCHS = 50
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = 'models/sign_language_model.h5'

os.makedirs('models', exist_ok=True)

def build_model(num_classes):
    # weights='imagenet': Sử dụng kiến thức đã học từ triệu ảnh thực tế
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Đóng băng các lớp dưới để giữ lại kiến thức cơ bản (feature extraction)
    base_model.trainable = False 
    
    # Xây dựng phần đầu mới (Classification Head)
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
    print("⏳ Đang load dữ liệu...")
    
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.exists(DATA_DIR):
        print(f"❌ Lỗi: Không tìm thấy thư mục '{DATA_DIR}'")
        return

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
    print(f"✅ Tìm thấy {len(class_names)} nhãn: {class_names}")
    
    with open('models/classes.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))

    # Tối ưu hiệu năng load dữ liệu (Data Pipeline Optimization)
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # --- 2. XÂY DỰNG VÀ TRAIN MODEL ---
    print("🏗️ Đang xây dựng model...")
    model = build_model(len(class_names))
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary() # In cấu trúc model để kiểm tra
    
    print("\n🚀 Bắt đầu Training trên GPU...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS
        )
        
        # --- 3. LƯU MODEL ---
        model.save(MODEL_SAVE_PATH)
        print(f"\n✅ Đã lưu model thành công tại: {MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình training: {e}")
        print("💡 Gợi ý: Nếu lỗi OOM (Out of Memory), hãy giảm BATCH_SIZE xuống 16 hoặc 8.")

if __name__ == "__main__":
    main()