from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow as tf


class FM(Model):
    def __init__(self, k, p):
        super(FM, self).__init__()
        self.w0 = tf.Variable(tf.zeros(1))
        self.w = tf.Variable(tf.zeros([p]))
        init_v = tf.random_normal_initializer(0, 0.1)
        self.v = tf.Variable(init_v([k, p]))

    def call(self, x):
        linear_terms = tf.add(self.w0, tf.reduce_sum(tf.multiply(self.w, x), 1, keepdims=True))
        pair_interactions = 0.5 * tf.reduce_sum(tf.subtract(tf.pow(tf.matmul(x, tf.transpose(self.v)), 2),
                                                            tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(self.v, 2)))),
                                                axis=1, keepdims=True)
        out = tf.add(linear_terms, pair_interactions)
        return out
