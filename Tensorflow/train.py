import os
from data import DataLoader
import tensorflow as tf
from loss import depth_loss_function
from model import Encoder, Decoder
from tensorflow import keras
from tensorflow.keras import Model, layers
import tensorflow.keras.backend as K


batch_size = 8
learning_rate = 0.0001
epochs = 50
save_lite = True

input_img = keras.Input(shape=(480, 640, 3))  # adapt this if using `channels_first` image data format
img_shape = keras.Input(shape=(2,),batch_size=1, name='sh',dtype='int32')
sh = K.mean(img_shape,axis=0)

encoder = Encoder()
decode_filters=int(encoder.layers[-1].output[0].shape[-1] // 2)
encoder = encoder(input_img)
decoder = Decoder(decode_filters=decode_filters,sh=sh)([encoder,sh])
autoencoder = Model([input_img,img_shape], decoder)
autoencoder.summary()


dl = DataLoader(DEBUG=False)
dataset = dl.get_batched_dataset(batch_size)

print('Data loader ready.')

optimizer = tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)

autoencoder.compile(loss=depth_loss_function, optimizer=optimizer)


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
autoencoder.fit(dataset, epochs=epochs, steps_per_epoch=dl.length // batch_size, shuffle=True)

#model.save("miomodel")
if save_lite:
    converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)
