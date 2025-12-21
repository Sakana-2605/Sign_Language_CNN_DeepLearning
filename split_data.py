import os
import shutil
import random

input_dir = "data_pipeline"
ouput_dir = "data_cnn"
SPLIT_RATIO = 0.8
SEED = 42

random.seed(SEED)

#create folders
for split in ["train", "val"]:
    for cls in os.listdir(input_dir):
        os.makedirs(os.path.join(ouput_dir, split, cls), exist_ok=True)

# split dataset
for cls in os.listdir(input_dir):
    imgs = os.listdir(os.path.join(input_dir, cls))
    random.shuffle(imgs)

    split_idx = int(len(imgs) * SPLIT_RATIO)
    train_imgs = imgs[:split_idx]
    val_imgs   = imgs[split_idx:]

    for img in train_imgs:
        shutil.copy(
            os.path.join(input_dir, cls, img),
            os.path.join(ouput_dir, "train", cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(input_dir, cls, img),
            os.path.join(ouput_dir, "val", cls, img)
        )

print("Done")
