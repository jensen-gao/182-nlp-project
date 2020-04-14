from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class CORAL(Layer):
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', **kwargs):
        super(CORAL, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], 1),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='zeros',
                                    trainable=True)
        super(CORAL, self).build(input_shape)

    def call(self, inputs, **kwargs):
        out = K.dot(inputs, self.kernel)
        out = K.tile(out, [1, self.output_dim])
        out = K.bias_add(out, self.bias)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim
