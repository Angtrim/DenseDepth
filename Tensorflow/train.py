import os
from data import DataLoader
import tensorflow as tf
from loss import depth_loss_function
from model import Encoder, Decoder
from tensorflow import keras
from tensorflow.keras import Model, layers
import tensorflow.keras.backend as K
import datetime
import numpy as np


batch_size = 8
learning_rate = 0.0001
epochs = 5
save_lite = True

input_img = keras.Input(shape=(480, 640, 3))  
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
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
autoencoder.fit(dataset, epochs=epochs, steps_per_epoch=dl.length // batch_size, shuffle=True, callbacks=[tensorboard_callback])

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']
input_shape2 = input_details[1]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_data2 = np.array([[input_data.shape[1]/32,input_data.shape[2]/32]],dtype="int32")
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.set_tensor(input_details[1]['index'], input_data2)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# Test the TensorFlow model on random input data.
tf_results = autoencoder([tf.constant(input_data),tf.constant(input_data2)])
print("Results shape")
print(tf_results.shape)
print(tflite_results.shape)

# Compare the result.
for tf_result, tflite_result in zip(tf_results, tflite_results):
  np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)
