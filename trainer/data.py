import cv2
import numpy as np
from tensorflow.python.lib.io import file_io
from trainer.constants import IM_SHAPE, IM_SHAPE_BBOX

TRAIN_FILE = './data/train.npz'
TEST_FILE = './data/test.npz'


def load_train_data(train_file_path=TRAIN_FILE, resize_shape=IM_SHAPE):
    train_file = file_io.FileIO(train_file_path, 'rb')
    data = np.load(train_file)
    images = resize_image_list(data['images'], resize_shape)
    masks = resize_image_list(data['masks'].astype(np.uint8), resize_shape)
    return images, masks


def load_test_data(test_file_path=TEST_FILE, resize_shape=IM_SHAPE):
    test_file = file_io.FileIO(test_file_path, 'rb')
    data = np.load(test_file)
    return resize_image_list(data['images'], resize_shape)


def resize_image_list(imgs, target_shape):
    return np.array([resize_image(img, target_shape) for img in imgs])


def resize_image(img, target_shape):
    return np.reshape(cv2.resize(img, dsize=target_shape[:2], interpolation=cv2.INTER_CUBIC), target_shape)


def augment(X, Y):
    X = np.append(X, [np.fliplr(x) for x in X], axis=0)
    Y = np.append(Y, [np.fliplr(y) for y in Y], axis=0)
    return X, Y


def bounding_box(masks):
    """
    Returns bounding box in format: [p_c, b_x, b_y, b_w, b_h], where each element
    denote whether there is a positive pixel, x-coordinate and y-coordinate of the
    center of the bbox, and height and width of the bbox, respectively.
    """
    bboxes = list()
    for i in range(len(masks)):
        mask = masks[i, ...].squeeze()
        rows = np.nonzero(np.any(mask, axis=1))[0]
        cols = np.nonzero(np.any(mask, axis=0))[0]
        if len(rows) == 0:
            # no positives
            bboxes.append([0, 0, 0, 0, 0])
            continue
        y_min = rows[0]
        y_max = rows[-1]
        x_min = cols[0]
        x_max = cols[-1]
        bboxes.append([1, (x_min + x_max) / 2, (y_min + y_max) / 2, (x_max - x_min), (y_max - y_min)])
    return np.array(bboxes)


def normalize_bounding_box(bboxes, scale=IM_SHAPE_BBOX):
    return bboxes / [1, scale[0], scale[1], scale[0], scale[1]]


def original_bounding_box(bboxes, scale=IM_SHAPE_BBOX):
    return bboxes * [1, scale[0], scale[1], scale[0], scale[1]]
