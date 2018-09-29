import unittest
import numpy as np
import tensorflow as tf

from trainer.data import normalize_bounding_box
from trainer.utils import bbox_loss_batch, bbox_mIOU, coord_only_mIOU
from tensorflow.python import debug as tf_debug


class TestLossAndMetrics(unittest.TestCase):
    def test_bbox_loss(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            loss_fn = bbox_loss_batch(5, 2, 1)
            labels = tf.placeholder(tf.float32, [None, 5], name='labels')
            predictions = tf.placeholder(tf.float32, [None, 5], name='predictions')
            loss = loss_fn(labels, predictions)
            sess.run(tf.global_variables_initializer())
            out = sess.run([loss], feed_dict={
                labels: [[1, 0.76, 0.54, 0.32, 0.1]],
                predictions: [[1, 60, 60, 100, 100]]
            })
            print(out)

    def test_bbox_mIOU(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'DESKTOP-QL1F43N:6064')
            labels = tf.placeholder(tf.float32, (None, 5), name='labels')
            predictions = tf.placeholder(tf.float32, [None, 5], name='predictions')
            iou = bbox_mIOU(labels, predictions)
            sess.run(tf.global_variables_initializer())
            out = sess.run([iou], feed_dict={
                labels: [[1, 60, 60, 100, 100]],
                predictions: [[0.5, 60, 60, 100, 100]]
            })
            print(out)
            # self.assertEqual(out[0], 0.5)

    def test_coord_only_mIOU(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'DESKTOP-QL1F43N:6064')
            labels = tf.placeholder(tf.float32, (None, 5), name='labels')
            predictions = tf.placeholder(tf.float32, [None, 5], name='predictions')
            iou = coord_only_mIOU(labels, predictions)
            sess.run(tf.global_variables_initializer())
            out = sess.run([iou], feed_dict={
                labels: [[1, 60, 60, 100, 100]],
                predictions: [[0.5, 60, 60, 100, 100]]
            })
            print(out)

    def test_normalize_bbox(self):
        bbox = np.array([1, 50, 40, 30, 20])
        scale = (100, 80)
        print(normalize_bounding_box(bbox, scale))


if __name__ == '__main__':
    t = TestLossAndMetrics()
    t.test_bbox_mIOU()
