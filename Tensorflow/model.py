from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.applications import DenseNet169
from up import BilinearUpSampling2D
import tensorflow.keras.backend as K
from tensorflow import keras
import tensorflow as tf



class UpscaleBlock(Model):
    def __init__(self, filters, size, name):
        super(UpscaleBlock, self).__init__()
        self.up = BilinearUpSampling2D(size=size, name=name + '_upsampling2d')
        self.concat = Concatenate(name=name + '_concat')  # Skip connection
        self.convA = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convA')
        self.reluA = LeakyReLU(alpha=0.2)
        self.convB = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convB')
        self.reluB = LeakyReLU(alpha=0.2)

    def call(self, x):
        upresult = self.up([x[1],x[0]])
        b = self.reluB(self.convB(self.reluA(self.convA(self.concat([upresult, x[2]])))))
        return b


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.base_model = DenseNet169(input_shape=(None, None, 3), include_top=False, weights='imagenet')
        print('Base model loaded {}'.format(DenseNet169.__name__))

        # Create encoder model that produce final features along with multiple intermediate features
        outputs = [self.base_model.outputs[-1]]
        for name in ['pool1', 'pool2_pool', 'pool3_pool', 'conv1/relu']: outputs.append(
            self.base_model.get_layer(name).output)
        self.all_outputs = outputs
        self.encoder = Model(inputs=self.base_model.inputs, outputs=outputs)

    def call(self, x):
        return self.encoder(x)


class Decoder(Model):
    def __init__(self, decode_filters):
        super(Decoder, self).__init__()
        self.conv2 = Conv2D(filters=decode_filters, kernel_size=1, padding='same', name='conv2')
        self.up1 = UpscaleBlock(filters=decode_filters // 2, size=(2, 2), name='up1')
        self.up2 = UpscaleBlock(filters=decode_filters // 4, size=(4, 4), name='up2')
        self.up3 = UpscaleBlock(filters=decode_filters // 8, size=(8, 8), name='up3')
        self.up4 = UpscaleBlock(filters=decode_filters // 16, size=(16, 16), name='up4')
        self.conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')


    def call(self, features):
        img_shape, x, pool1, pool2, pool3, conv1 = features[1], features[0][0], features[0][1], features[0][2], features[0][3], features[0][4]
        # Get image shape dynamically by input and divide in by 32 (Max upscale in net)
        sh = K.mean(img_shape, axis=0)
        up0 = self.conv2(x)
        up1 = self.up1([up0, sh, pool3])
        up2 = self.up2([up1, sh, pool2])
        up3 = self.up3([up2, sh, pool1])
        up4 = self.up4([up3, sh, conv1])
        return self.conv3(up4)





