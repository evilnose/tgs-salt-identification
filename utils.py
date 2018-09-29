import numpy as np
import tensorflow as tf
import keras.backend as K


def mIOU(Y_true, Y_pred):
    precisions = list()
    for threshold in np.arange(0.5, 1, 0.05):
        Y_pred_threshold = tf.to_int32(Y_pred > threshold)
        score, up_opt = tf.metrics.mean_iou(Y_true, Y_pred_threshold, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        precisions.append(score)
    return K.mean(K.stack(precisions), axis=0)


def config_keras():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    K.set_session(sess)  # set this TensorFlow session as the default session for Keras
