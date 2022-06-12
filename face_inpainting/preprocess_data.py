import numpy as np
import tensorflow as tf
import h5py

from constants import *


def get_data():
    h5f = h5py.File(HDF5_PATH + '/dataset.hdf5', 'r')
    unmasked = h5f['celeba'][0:DATASET_SIZE]
    masked = h5f['masked_faces'][0:DATASET_SIZE]
    # segmaps = h5f['segmentation_masks'][0:DATASET_SIZE]
    segmap_file = h5py.File(HDF5_PATH + '/predicted_segmaps.hdf5')
    segmaps = segmap_file['segmaps'][0:DATASET_SIZE]
    h5f.close()
    segmap_file.close()

    unmasked = np.array(unmasked)
    masked = np.array(masked)
    segmaps = np.array(segmaps)

    inverse_segmaps = (segmaps - 1) / 255  # uint8 arithmetic

    removed_masks = []

    for i in range(DATASET_SIZE):
        removed_masks.append(masked[i] * (np.repeat(np.expand_dims(inverse_segmaps[i], axis=-1), 3, axis=-1)) / 255)

    return masked, segmaps, unmasked, inverse_segmaps, removed_masks


def get_xy():
    masked, segmaps, unmasked, inverse_segmaps, removed_masks = get_data()

    x = tf.convert_to_tensor(removed_masks)
    y = tf.convert_to_tensor(np.array(unmasked) / 255)

    return x, y


def get_datasets():
    x, y = get_xy()

    train_data = tf.data.Dataset.from_tensors((x[:test_start], y[:test_start]))
    test_data = tf.data.Dataset.from_tensors((x[test_start:], y[test_start:]))

    return train_data, test_data
