import os
from model import DepthEstimate
from data import DataLoader
import tensorflow as tf
from loss import depth_loss_function

batch_size = 8
learning_rate = 0.0001
epochs = 1


model = DepthEstimate()
dl = DataLoader(DEBUG=True)
dataset = dl.get_batched_dataset(batch_size)

print('Data loader ready.')

optimizer = tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)

model.compile(loss=depth_loss_function, optimizer=optimizer)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
model.fit(dataset, epochs=epochs, steps_per_epoch=dl.length // batch_size, shuffle=True, callbacks=[cp_callback])
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)
