import tensorflow as tf
from pathlib import Path

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS_1 = 15   # train head
EPOCHS_2 = 10   # fine-tune
input_dir = "data_cnn"


def build_image_dataset_from_directory(directory, image_size=(128, 128), batch_size=32, shuffle=True):
    p = Path(directory)
    if not p.exists():
        raise ValueError(f"Directory not found: {directory}")

    # collect class directories in deterministic order
    class_dirs = [d for d in sorted(p.iterdir()) if d.is_dir()]
    class_names = [d.name for d in class_dirs]

    file_paths = []
    labels = []
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    for idx, d in enumerate(class_dirs):
        for ext in exts:
            for f in d.glob(ext):
                file_paths.append(str(f))
                labels.append(idx)

    if len(file_paths) == 0:
        raise ValueError(f"No image files found under {directory}")

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths))

    def _process(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        return image, label

    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # attach class_names attribute so existing code can read it
    ds.class_names = class_names
    return ds


train_ds = build_image_dataset_from_directory(f"{input_dir}/train", image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

val_ds = build_image_dataset_from_directory(f"{input_dir}/val", image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(train_ds.class_names)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("🚀 Training head...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_1
)

print("🔧 Fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_2
)

model.save("model.h5")
print("Model saved as model.h5")
