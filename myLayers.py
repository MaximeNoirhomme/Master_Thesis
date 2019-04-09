from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
class BinMultiplexerLayer(Layer):
    def call(self, x):
        '''
            Select
        '''

        domain_input = x[2]

        is_sources = tf.cast(domain_input, tf.bool)
        is_sources = tf.tile(is_sources, [0, tf.shape(x[0])[1]])

        return tf.where(is_sources, x[1], x[0])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class KroneckerProductLayer(Layer):
    def call(self, x):
        shape_f = tf.shape(x[0])
        shape_s = tf.shape(x[1])

        tmp = tf.reshape(x[0], [shape_f[0], shape_f[1], 1], name='tmp1') * tf.reshape(x[1], [shape_s[0], 1,shape_s[1]], name='tmp2')
        
        return tf.reshape(tmp,[shape_f[0],shape_f[1]*shape_s[1]], name='final_reshape')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1] * input_shape[1][1])