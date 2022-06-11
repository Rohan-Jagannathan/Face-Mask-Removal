import tensorflow as tf
import numpy as np
import h5py

from constants import *


def get_data():
    """Initialize Input and Segmap Arrays from Pickle"""

    # read images from HDF5 into numpy arrays
    h5f = h5py.File(HDF5_PATH, 'r')
    inputs = h5f['masked_faces'][0:DATASET_SIZE]
    segmaps = h5f['segmentation_masks'][0:DATASET_SIZE]
    h5f.close()

    return inputs, segmaps


def get_data_loaders():
    inputs, segmaps = get_data()

    x = tf.convert_to_tensor(np.array(inputs) / 255)
    y = tf.one_hot(segmaps, 2, dtype='uint8') if ONE_HOT_ENCODE else segmaps

    train_data = tf.data.Dataset.from_tensors((x[:train_end], y[:train_end]))
    valid_data = tf.data.Dataset.from_tensors((x[train_end:valid_end], y[train_end:valid_end]))
    test_data = tf.data.Dataset.from_tensors((x[valid_end:], y[valid_end:]))

    return train_data, valid_data, test_data
