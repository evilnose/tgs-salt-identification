import os
import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
import cv2
import constants

TRAIN_DIR = './data/train/'
TEST_DIR = './data/test/'


def load_train_data(im_shape=constants.IM_SHAPE):
    img_dir = os.path.join(TRAIN_DIR, 'images')
    mask_dir = os.path.join(TRAIN_DIR, 'masks')
    ids = next(os.walk(img_dir))[2]

    m = len(ids)
    X_train = np.zeros(shape=(m,) + im_shape)
    Y_train = np.zeros(shape=(m,) + im_shape, dtype=np.bool)

    for i, id_ in tqdm(enumerate(ids), total=m):
        img = mpimg.imread(os.path.join(img_dir, id_))[:, :, 0]
        X_train[i] = resize_image(img, im_shape)
        mask = mpimg.imread(os.path.join(mask_dir, id_))
        Y_train[i] = resize_image(mask, im_shape)

    return X_train, Y_train


def resize_image(img, target_shape):
    return np.reshape(cv2.resize(img, dsize=target_shape[:2], interpolation=cv2.INTER_CUBIC), target_shape)


def augment(X, Y):
    X = np.append(X, [np.fliplr(x) for x in X], axis=0)
    Y = np.append(Y, [np.fliplr(y) for y in Y], axis=0)
    return X, Y
