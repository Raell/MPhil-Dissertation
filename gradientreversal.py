import tensorflow as tf
from tensorflow.keras.layers import Layer

@tf.custom_gradient
def grad_reverse(y, hp_lambda):
    def custom_grad(dy):
        return -dy * hp_lambda
    return y, custom_grad


class GradientReversal(Layer):
    def __init__(self, hp_lambda):
        super(GradientReversal, self).__init__()
        self.hp_lambda = hp_lambda

    def call(self, x):
        return grad_reverse(x, self.hp_lambda)