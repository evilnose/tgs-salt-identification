import numpy as np
import os
import matplotlib.image as mpimg
from tqdm import tqdm

from trainer.constants import IM_SHAPE_RAW
from trainer.data import resize_image

RAW_TRAIN_DIR = 'raw_data/train'
RAW_TEST_DIR = 'raw_data/test'


def load_train_data_raw(im_shape=IM_SHAPE_RAW):
    img_dir = os.path.join(RAW_TRAIN_DIR, 'images')
    mask_dir = os.path.join(RAW_TRAIN_DIR, 'masks')
    ids = next(os.walk(img_dir))[2]

    m = len(ids)
    X_train = np.zeros(shape=(m,) + im_shape)
    Y_train = np.zeros(shape=(m,) + im_shape, dtype=np.bool)

    for i, id_ in tqdm(enumerate(ids), total=m):
        X_train[i] = mpimg.imread(os.path.join(img_dir, id_))[:, :, 0]
        Y_train[i] = mpimg.imread(os.path.join(mask_dir, id_))

    return X_train, Y_train


def load_test_data_raw(im_shape=IM_SHAPE_RAW):
    img_dir = os.path.join(RAW_TRAIN_DIR, 'images')
    ids = next(os.walk(img_dir))[2]

    m = len(ids)
    X_test = np.zeros(shape=(m,) + im_shape)
    for i, id_ in tqdm(enumerate(ids), total=m):
        img = mpimg.imread(os.path.join(img_dir, id_))[:, :, 0]
        X_test[i] = img

    return X_test


def save_train_arr():
    images, masks = load_train_data_raw()
    np.savez_compressed('data/train', images=images, masks=masks)
    print("Saved.")


def save_test_arr():
    images = load_test_data_raw()
    np.savez_compressed('data/test', images=images)
    print("Saved.")


if __name__ == '__main__':
    save_test_arr()
