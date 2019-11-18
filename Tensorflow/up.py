from tensorflow.keras.layers  import Layer, InputSpec
import keras.utils.conv_utils as conv_utils
import tensorflow as tf
import tensorflow.keras.backend as K
import keras.backend as K2

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = K2.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs):
        input_sh = inputs[0]
        
        height = self.size[0] * input_sh[0] if input_sh[0] is not None else None
        width = self.size[1] * input_sh[1] if input_sh[1] is not None else None

        return tf.image.resize(inputs[1], [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
