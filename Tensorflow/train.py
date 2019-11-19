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
import utils


batch_size = 8
learning_rate = 0.0001
epochs = 1
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

# Load training data
train_data_loader = DataLoader(DEBUG=True)
train_dataset = train_data_loader.get_batched_dataset(batch_size)

# Load validation data
val_data_loader = DataLoader(DEBUG=True, csv_file='data/nyu2_test.csv')
val_dataset = val_data_loader.get_batched_dataset(batch_size)

# Compile model
optimizer = tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)
autoencoder.compile(loss=depth_loss_function, optimizer=optimizer)

# Create training folder
training_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
training_folder = "training_"+training_id
checkpoint_path = training_folder+"/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
autoencoder.fit(train_dataset, epochs=epochs, steps_per_epoch=train_data_loader.length // batch_size, shuffle=True, callbacks=[cp_callback], validation_data=val_dataset, validation_steps=val_data_loader.length // batch_size)

# Convert the model.
tf_lite_model = utils.save_lite_model(autoencoder, training_folder, "model.tflite")

# Load an image
image = np.array(np.random.random_sample( (480, 640, 3)), dtype=np.float32)

# Run lite and standard
tflite_results = utils.run_lite_model(image, tf_lite_model)
tf_results = utils.run_std_model(image, autoencoder)

# Compare the result.
for tf_result, tflite_result in zip(tf_results[0], tflite_results[0]):
  np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)




