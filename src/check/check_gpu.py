import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ ĐÃ NHẬN GPU: {gpus}")
    print("Chi tiết:", tf.config.experimental.get_device_details(gpus[0]))
else:
    print("❌ CHƯA NHẬN GPU (Đang chạy CPU)")