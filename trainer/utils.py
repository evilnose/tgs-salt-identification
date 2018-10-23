import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.python.lib.io import file_io


def mIOU(y_true, y_pred):
    precisions = list()
    for threshold in np.arange(0.5, 1, 0.05):
        y_pred_t = tf.to_int32(y_pred > threshold)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_t, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        precisions.append(score)
    return K.mean(K.stack(precisions), axis=0)


def real_mIOU(y_true, y_pred):
    y_true = y_true.astype(np.bool)
    precisions = list()
    for th in np.arange(0.5, 1, 0.05):
        y_pred_th = (y_pred > th)
        intersection = y_pred_th * y_true
        union = y_pred_th + y_true
        precisions.append(intersection.sum() / union.sum())
    return np.mean(np.stack(precisions), axis=0)


def confusion(y_true, y_pred):
    y_true = y_true.astype(np.bool)
    results = np.zeros(4)
    n = 0
    y_false = ~y_true
    for th in np.arange(0.5, 1, 0.05):
        y_pred_th = (y_pred > th)
        y_pred_th_false = ~y_pred_th
        results[0] += (y_pred_th * y_true).sum()
        results[1] += (y_pred_th_false * ~y_true).sum()
        results[2] += (y_pred_th * ~y_true).sum()
        results[3] += (y_pred_th_false * y_true).sum()
        n += 1
    return results / n


def bbox_mIOU(Y_true, Y_pred):
    elems = (Y_true, Y_pred)
    results = tf.map_fn(lambda elem: __bbox_mIOU_single(elem[0], elem[1]), elems, dtype=tf.float32)
    # Get all results greater than zero (valid)
    valid_results = tf.boolean_mask(results, tf.greater_equal(results, 0))
    # If not valid results exist, return 0; else return mean
    return tf.where(tf.equal(tf.size(valid_results), 0), 0., tf.reduce_mean(valid_results))


def __bbox_mIOU_single(y_true, y_pred):
    precision_sum = tf.zeros((1,), dtype=tf.float32)
    num_valid = tf.Variable(0, dtype=tf.uint8)
    for threshold in np.arange(0.5, 1, 0.05):
        pred_truth = tf.greater_equal(y_pred[0], threshold)
        iou = tf.cond(pred_truth,
                      lambda: __bbox_iou_pure(y_pred[1:], y_true[1:]),
                      lambda: __bbox_iou_pure(tf.zeros(4), y_true[1:]),
                      )
        isnan = tf.is_nan(iou)
        num_valid = tf.where(isnan, num_valid, num_valid + 1)
        precision_sum = tf.where(isnan, precision_sum, precision_sum + iou)

    # return -1 to flag result as invalid
    return tf.where(tf.equal(num_valid, 0), tf.constant(-1, dtype=tf.float32, shape=[1]),
                    precision_sum / tf.to_float(num_valid))


def __bbox_iou_pure(box1_wh, box2_wh):
    def to_coord(box):
        """Returns the lower-left and upper-right coordinates of the bounding box specified by box {x, y, w, h}"""
        half_w = tf.maximum(box[2] / 2, 0.)
        half_h = tf.maximum(box[3] / 2, 0.)
        res = [box[0] - half_w, box[1] - half_h, box[0] + half_w, box[1] + half_h]
        return res

    box1 = to_coord(box1_wh)
    box2 = to_coord(box2_wh)

    # Taken from Coursera assignment
    xi1 = tf.maximum(box1[0], box2[0])
    yi1 = tf.maximum(box1[1], box2[1])
    xi2 = tf.minimum(box1[2], box2[2])
    yi2 = tf.minimum(box1[3], box2[3])
    inter_area = tf.maximum(xi2 - xi1, 0.) * tf.maximum(yi2 - yi1, 0.)

    box1_area = tf.maximum(box1_wh[2], 0.) * tf.maximum(box1_wh[3], 0.)
    box2_area = tf.maximum(box2_wh[2], 0.) * tf.maximum(box2_wh[3], 0.)
    union_area = (box1_area + box2_area) - inter_area

    iou = inter_area / union_area
    return iou


def bbox_loss_batch(coord_scale, obj_scale, noobj_scale):
    def weighted_mse(y1, y2, weights):
        return tf.losses.mean_squared_error(y1, y2, weights=weights)

    def signed_sqrt(x):
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def inner_loss(Y_true, Y_pred):
        """Factory-produced loss function. Similar to YOLO's loss function"""
        Y_obj_exists = tf.not_equal(Y_true[:, 0], 0.)
        # Penalize x and y predictions
        xy_loss = sum(weighted_mse(Y_true[:, i], Y_pred[:, i], Y_obj_exists) for i in [1, 2])
        # Penalize w and h predictions
        wh_loss = sum(weighted_mse(tf.sqrt(Y_true[:, i]), signed_sqrt(Y_pred[:, i]), Y_obj_exists) for i in [3, 4])
        # Penalize false positive
        fp_loss = weighted_mse(Y_true[:, 0], Y_pred[:, 0], weights=Y_obj_exists)
        # Penalize false negative
        fn_loss = weighted_mse(Y_true[:, 0], Y_pred[:, 0], weights=tf.logical_not(Y_obj_exists))

        return coord_scale * (xy_loss + wh_loss) + obj_scale * fp_loss + noobj_scale * fn_loss

    return inner_loss


def binary_mse(y_true, y_pred):
    return tf.losses.mean_squared_error(y_true[:, 0], y_pred[:, 0])


def coord_only_mIOU(Y_true, Y_pred):
    elems = (Y_true, Y_pred)
    results = tf.map_fn(lambda elem: __bbox_mIOU_single(elem[0], elem[1]), elems, dtype=tf.float32)
    # Get all results greater than zero (valid)
    valid_results = tf.boolean_mask(results, tf.greater_equal(results, 0))
    # If not valid results exist, return 0; else return mean
    return tf.where(tf.equal(tf.size(valid_results), 0), 0., tf.reduce_mean(valid_results))


def __coord_only_mIOU_single(y_true, y_pred):
    return tf.cond(tf.equal(y_true[0], 1), lambda: __bbox_iou_pure(y_pred[1:], y_true[1:]), lambda: -1)


def config_keras():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    K.set_session(sess)  # set this TensorFlow session as the default session for Keras


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


class ModelSave(ModelCheckpoint):
    """Continuous eval callback to evaluate the checkpoint once
       every so many epochs.
    """

    def __init__(self, filepath, target_dir, **kwargs):
        super().__init__(filepath, **kwargs)
        self.target_dir = target_dir
        if self.filepath.startswith('gs://'):
            self.saving_to_gcs = True
        else:
            self.saving_to_gcs = False

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.saving_to_gcs:
            copy_file_to_gcs(self.target_dir, self.filepath)
