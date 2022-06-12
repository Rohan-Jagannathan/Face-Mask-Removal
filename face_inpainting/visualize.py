import matplotlib.pyplot as plt
import numpy as np
from random import randint
import visualkeras

from preprocess_data import get_data
from constants import *
from autoencoder import get_model


def view_data(index):
    masked, segmaps, unmasked, inverse_segmaps, removed_masks = get_data()

    figure = plt.figure(figsize=(10, 10))

    unmasked_img = figure.add_subplot(231)
    plt.imshow(unmasked[index])

    masked_img = figure.add_subplot(232)
    plt.imshow(masked[index])

    removed = figure.add_subplot(233)
    plt.imshow(
        (masked[index] / 255) * (np.repeat(np.expand_dims(inverse_segmaps[index], axis=-1), 3, axis=-1)))

    segmap = figure.add_subplot(234)
    plt.imshow(segmaps[index], 'Greys_r')

    inverse = figure.add_subplot(235)
    plt.imshow(inverse_segmaps[index], 'Greys_r')

    unmasked_img.title.set_text('Original Image')
    masked_img.title.set_text('Masked Image')
    removed.title.set_text('Removed Mask')
    segmap.title.set_text('Predicted Segmentation Map')
    inverse.title.set_text('Inverse Segmentation Map')

    plt.show()


def view_model():
    model = get_model()
    plt.imshow(visualkeras.layered_view(model, legend=True))
    plt.show()


if __name__ == '__main__':
    view_data(randint(0, DATASET_SIZE))
    view_model()
