import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from conf import CLASSES, INPUT_SHAPE, TRAIN_DATA, VAL_DATA


class DataLoader:
    def __init__(self, mode, sample_type):
        """
        :param mode: Determine whether the data is being used for training, evaluation or prediction
        :param sample_type: Select the type of data to be used for training
        :return: Nothing
        :doc-author: Trelent
        """
        self.mode = mode
        self.sample_type = sample_type
        assert self.mode in ('train', 'eval', 'predict')

    def load_image(self, image_path):
        """
        The load_image function is used to load the image data from the .npy files.
        The function takes in an image path and returns a tensor of shape INPUT_SHAPE.
        If sample_type is 'all', then it loads all three types of samples: spects, mfccs and chromas.
        Otherwise, it only loads one type of sample.

        :param image_path: Load the image from a path
        :return: A tensor of shape INPUT_SHAPE
        :doc-author: Trelent
        """
        image_path = image_path.numpy().replace(b".png", b".npy")
        image, image2, image3 = None, None, None
        if self.sample_type in ("spects", "mfccs", "chromas"):
            sample_type = bytes(f'{self.sample_type}', 'utf-8')
            image = np.load(image_path.replace(b"images", sample_type))
            image = tf.image.resize(image[..., tf.newaxis], INPUT_SHAPE[:2], method=ResizeMethod.NEAREST_NEIGHBOR)
            image = tf.concat([image, image, image], axis=-1) * 255.
        elif self.sample_type in ('rgb', 'all'):
            image = np.load(image_path.replace(b"images", b"spects"))  # spects
            image2 = np.load(image_path.replace(b"images", b"mfccs"))  # mfccs
            image3 = np.load(image_path.replace(b"images", b"chromas"))  # chromas
            image = tf.image.resize(image[..., tf.newaxis], INPUT_SHAPE[:2], method=ResizeMethod.NEAREST_NEIGHBOR)
            image2 = tf.image.resize(image2[..., tf.newaxis], INPUT_SHAPE[:2], method=ResizeMethod.NEAREST_NEIGHBOR)
            image3 = tf.image.resize(image3[..., tf.newaxis], INPUT_SHAPE[:2], method=ResizeMethod.NEAREST_NEIGHBOR)
            if self.sample_type == 'rgb':
                image = tf.concat([image, image2, image3], axis=-1) * 255.
            else:
                image = tf.concat([image, image, image], axis=-1) * 255.
                image2 = tf.concat([image2, image2, image2], axis=-1) * 255.
                image3 = tf.concat([image3, image3, image3], axis=-1) * 255.
        if self.sample_type == 'all':
            return image, image2, image3
        else:
            return image

    def load_df(self):
        """
        The load_df function loads the dataframe from a csv file.

        :return: A dataframe
        :doc-author: Trelent
        """
        if self.mode == 'train':
            file = TRAIN_DATA
        elif self.mode == 'eval':
            file = VAL_DATA
        else:
            file = VAL_DATA
        return pd.read_csv(file)

    def load_ds(self, batch_size):
        """
        The load_ds function is responsible for loading the data from disk and preparing it for training.
        It does this by:
            1) Loading the CSV file containing image paths and class labels into a Pandas DataFrame.
            2) Creating two TensorFlow Datasets, one for images and one for class labels (if in train or eval mode).
               The datasets are then zipped together to create a single dataset of tuples containing an image path
               string and its corresponding label. This dataset is shuffled if in train mode,
               but not if in eval or predict modes.

        :param batch_size: Determine the number of images to load in a batch
        :return: A dataset object
        :doc-author: Trelent
        """
        df = self.load_df()
        ds_image = tf.data.Dataset.from_tensor_slices(np.asarray(df['image'].values).astype(np.str))

        if self.mode != 'predict':
            ds_target = tf.data.Dataset.from_tensor_slices(np.asarray(df['class'].values).astype(np.int32))
            ds = tf.data.Dataset.zip((ds_image, ds_target))
            if self.mode == 'train':
                ds = ds.shuffle(buffer_size=ds.cardinality())
            if self.sample_type == 'all':
                ds = ds.map(name='Load_Images',
                            map_func=lambda img, cls:
                            (tf.py_function(self.load_image, inp=img, Tout=[tf.float32, tf.float32, tf.float32]),
                             tf.one_hot(cls, len(CLASSES))),
                            num_parallel_calls=tf.data.AUTOTUNE)
                ds = ds.map(map_func=lambda imgs, cls:
                            ({'spects': imgs[0], 'mfccs': imgs[1], 'chromas': imgs[2]},
                             {'class': cls}),
                            num_parallel_calls=tf.data.AUTOTUNE)
            else:
                ds = ds.map(name='Load_Images',
                            map_func=lambda img, cls:
                            ({'inputs': tf.py_function(self.load_image, inp=[img], Tout=tf.float32)},
                             {'class': tf.one_hot(cls, len(CLASSES))}),
                            num_parallel_calls=tf.data.AUTOTUNE)
        else:
            if self.sample_type == 'all':
                ds = ds_image.map(name='Load_Images',
                                  map_func=lambda img:
                                  (tf.py_function(self.load_image, inp=[img], Tout=[tf.float32, tf.float32, tf.float32])),
                                  num_parallel_calls=tf.data.AUTOTUNE)
                ds = ds.map(map_func=lambda spects, mfccs, chromas: {'spects': spects, 'mfccs': mfccs, 'chromas': chromas},
                            num_parallel_calls=tf.data.AUTOTUNE)
            else:
                ds = ds_image.map(name='Load_Images',
                                  map_func=lambda img:
                                  {'inputs': tf.py_function(self.load_image, inp=[img], Tout=tf.float32)},
                                  num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(batch_size=batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


if __name__ == '__main__':
    exit()
