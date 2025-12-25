import os
import tensorflow as tf
from tensorflow.keras.applications import EfficiencyNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
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

# --- 2. CẤU HÌNH PARAMETERS ---
DATA_DIR = './data_augmented' 
IMG_SIZE = (224, 224) # Kích thước chuẩn tối ưu cho EfficiencyNet-B0
BATCH_SIZE = 32       # Nếu bị lỗi OOM, hãy giảm xuống 16
EPOCHS = 50
LEARNING_RATE = 0.001 # Bắt đầu với LR cao hơn một chút vì ta dùng ReduceLROnPlateau
MODEL_SAVE_PATH = 'models/sign_language_effnet.h5'

os.makedirs('models', exist_ok=True)

def build_model(num_classes):
    # Khởi tạo base model với trọng số ImageNet
    base_model = EfficiencyNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Bước đầu: Đóng băng base model
    base_model.trainable = False 
    
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Lưu ý: EfficiencyNetB0 ĐÃ TÍCH HỢP sẵn lớp Rescaling (0-255 -> 0-1) bên trong. 
    # Không cần preprocess_input như MobileNetV2.
    x = base_model(inputs, training=False)
    
    x = GlobalAveragePooling2D()(x) 
    x = Dropout(0.3)(x) # Tăng dropout một chút để tránh overfitting
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x) 
    
    model = Model(inputs, outputs)
    return model

def main():
    # --- 3. LOAD DỮ LIỆU ---
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    # Lưu nhãn để dùng sau này
    with open('models/classes.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))

    # Tối ưu Pipeline
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # --- 4. CALLBACKS (BỘ NÃO CỦA QUÁ TRÌNH TRAIN) ---
    callbacks = [
        # Dừng nếu val_loss không giảm sau 6 epoch
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        # Giảm LR nếu model bị chững lại (giúp hội tụ sâu hơn)
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        # Lưu model tốt nhất trong quá trình chạy
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)
    ]

    # --- 5. TRAINING ---
    model = build_model(num_classes)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("\n🚀 Giai đoạn 1: Train lớp phân loại cuối cùng...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # --- 6. FINE-TUNING (NÂNG CAO) ---
    print("\n🚀 Giai đoạn 2: Fine-tuning một phần Base Model...")
    # Mở khóa toàn bộ model
    for layer in model.layers:
        if isinstance(layer, Model): # Chính là base_model
            layer.trainable = True
            # Đóng băng lại các lớp đầu (ví dụ: chỉ mở 30 lớp cuối)
            for l in layer.layers[:-30]:
                l.trainable = False

    # Compile lại với LR cực nhỏ
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

    print(f"\n✅ Hoàn tất! Model tốt nhất đã được lưu tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()