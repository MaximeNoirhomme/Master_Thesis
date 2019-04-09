import tensorflow as tf
from keras.engine import Layer
import keras.backend as K
from math import exp

'''
    This code does not belong to me: https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
    GradientReversal has been modified in order to use non-constant lambda parameters.
'''


def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, l0, gamma, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.q = 0.1
        self.l0 = l0
        self.gamma = gamma
        #self.epoch = epoch
        self.hp_lambda = self._compute_lambda() #hp_lambda
        print("q = ", self.q)
        print("hp_lambda = ", self.hp_lambda)

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def update_lambda(self, incr):
        self.q += incr
        self.hp_lambda = self._compute_lambda()
        print("q = ", self.q)
        print("hp_lambda = ", self.hp_lambda)

    def _compute_lambda(self):
        #return self.l0 * (2 / (1 + exp(- self.gamma * 1)) - 1)
        return self.l0 * (2 / (1 + exp(- self.gamma * self.q)) - 1)
