import tensorflow as tf

if __name__ == '__main__':
    Y_true = tf.constant([[1, 2, 3, 4, 5]], name='y_true')
    Y_pred = tf.constant([[1, 2, 3, 4, 5]], name='y_pred')
    # with tf.Session() as sess:
