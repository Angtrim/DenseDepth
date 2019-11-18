from keras.layers  import Layer, InputSpec
import keras.utils.conv_utils as conv_utils
import tensorflow as tf
import keras.backend as K

class BilinearUpSampling2D(Layer):
    def __init__(self, input_sh, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_sh = input_sh

    def compute_output_shape(self, input_shapes):
        input_img = input_shapes[1]
        input_shape = input_shapes[0]
        print(input_shape)


        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[0] if input_shape[0] is not None else None
            width = self.size[1] * input_shape[1] if input_shape[1] is not None else None
            return (input_img[0],
                    input_img[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[0] if input_shape[0] is not None else None
            width = self.size[1] * input_shape[0] if input_shape[0] is not None else None
            return (input_img[0],
                    None,
                    None,
                    input_img[3])

    def call(self, inputs):
        input_sh = inputs[0]
        if self.data_format == 'channels_first':
            height = self.size[0] * input_sh[0] if input_sh[0] is not None else None
            width = self.size[1] * input_sh[1] if input_sh[1] is not None else None
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_sh[0] if input_sh[0] is not None else None
            width = self.size[1] * input_sh[1] if input_sh[1] is not None else None
        return tf.image.resize(inputs[1], [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
