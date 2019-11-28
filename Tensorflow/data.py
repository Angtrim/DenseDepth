import tensorflow as tf
import random
from io import BytesIO
from zipfile import ZipFile
from sklearn.utils import shuffle
import numpy as np

class DataLoader():
    def __init__(self, csv_file='data/nyu2_train.csv', DEBUG=False):
        self.shape_rgb = (480, 640, 3)
        self.shape_depth = (240, 320, 1)
        self.read_nyu_data(csv_file, DEBUG=DEBUG)

    def nyu_resize(self, img, resolution=480, padding=6):
        from skimage.transform import resize
        return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

    def read_nyu_data(self, csv_file, DEBUG=False):
        csv = open(csv_file, 'r').read()
        nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))

        # Dataset shuffling happens here
        #nyu2_train = shuffle(nyu2_train, random_state=0)

        # Test on a smaller dataset
        if DEBUG: nyu2_train = nyu2_train[:10]
        
        # A vector of RGB filenames.
        self.filenames = [i[0] for i in nyu2_train]

        # A vector of depth filenames.
        self.labels = [i[1] for i in nyu2_train]

        # Length of dataset
        self.length = len(self.filenames)

    def _parse_function(self, filename, label): 
        # Read images from disk
        image_decoded = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
        depth_decoded = tf.image.decode_jpeg(tf.io.read_file(label))

        # Format
        rgb = tf.image.convert_image_dtype(image_decoded, tf.float32)
        depth = tf.image.convert_image_dtype(depth_decoded, dtype=tf.float32)
        depth = tf.image.resize(depth, [self.shape_depth[0],self.shape_depth[1]])

        # p = random.random()
        # if p > 0.5:
        #     rgb, depth = self.augment(rgb, depth)

        
        # Normalize the depth values (in cm)
        #depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)
        #depth = 1.0 / tf.clip_by_value(depth, 0, 1)

        # depth = depth - tf.reduce_min(depth)
        # depth = depth/tf.reduce_max(depth)

        # depth = ((1/(1 + depth)) - 0.5)*2

        img_size = np.array([self.shape_rgb[0], self.shape_rgb[1]])
        data_tf = tf.convert_to_tensor(img_size, np.int32)

        return (rgb, data_tf), depth

    def augment(self, rgb, depth): 
        # Read images from disk
        p = random.random()
        if p > 0.2:
            rgb = tf.image.flip_left_right(rgb)
            depth = tf.image.flip_left_right(depth)

        p = random.random()
        if p > 0.5:
            rgb = tf.image.adjust_brightness(rgb, 0.2)

        p = random.random()
        if p > 0.5:
            rgb = tf.image.adjust_contrast(rgb, 0.6)

        p = random.random()
        if p > 0.5:
            rgb = tf.image.adjust_saturation(rgb, 0.6)

        return rgb, depth

    def get_batched_dataset(self, batch_size):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
        self.dataset = self.dataset.shuffle(self.length)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.map(map_func=self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.batch(batch_size=batch_size)
        return self.dataset

