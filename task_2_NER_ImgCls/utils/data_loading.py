import os
import shutil
from sklearn.model_selection import train_test_split
from task_2_NER_ImgCls.src.paths import TRAIN_DIR, TEST_DIR, VAL_DIR

raw_img_dir = 'path/to/raw_imgs'  # path to downloaded data

train_dir = TRAIN_DIR
val_dir = TEST_DIR
test_dir = VAL_DIR

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


def split_data(raw_img_dir, train_dir, val_dir, test_dir, test_size=0.15, val_size=0.15):
    for animal_folder in os.listdir(raw_img_dir):
        animal_folder_path = os.path.join(raw_img_dir, animal_folder)

        if os.path.isdir(animal_folder_path):
            train_animal_dir = os.path.join(train_dir, animal_folder)
            val_animal_dir = os.path.join(val_dir, animal_folder)
            test_animal_dir = os.path.join(test_dir, animal_folder)

            os.makedirs(train_animal_dir, exist_ok=True)
            os.makedirs(val_animal_dir, exist_ok=True)
            os.makedirs(test_animal_dir, exist_ok=True)

            images = [f for f in os.listdir(animal_folder_path) if os.path.isfile(os.path.join(animal_folder_path, f))]

            train_images, test_val_images = train_test_split(images, test_size=test_size + val_size)
            val_images, test_images = train_test_split(test_val_images, test_size=test_size / (test_size + val_size))

            for img in train_images:
                shutil.copy(os.path.join(animal_folder_path, img), os.path.join(train_animal_dir, img))
            for img in val_images:
                shutil.copy(os.path.join(animal_folder_path, img), os.path.join(val_animal_dir, img))
            for img in test_images:
                shutil.copy(os.path.join(animal_folder_path, img), os.path.join(test_animal_dir, img))


split_data(raw_img_dir, train_dir, val_dir, test_dir)
