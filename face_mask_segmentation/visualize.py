import matplotlib.pyplot as plt
from random import randint

from initialize_data import *


def show_sample_image():
    """Show Input-Label Pair Example"""
    inputs, segmaps = get_data()

    rand_index = randint(0, DATASET_SIZE)
    plt.imshow(inputs[rand_index])
    plt.show()

    plt.imshow(segmaps[rand_index])
    plt.show()


def show_histogram():
    """Segmentation Classes Histogram"""
    inputs, segmaps = get_data()

    values = [0] * 2
    names = ['not mask', 'mask']
    colors = [(0, 0, 0), (1, 1, 0)]

    for ctr, segmap in enumerate(segmaps):
        for i in range(0, len(segmap) - 1):
            for j in range(0, len(segmap[i]) - 1):
                values[segmap[i][j]] += 1

    plt.title('Segmentation Distribution')
    plt.bar(names, values, color=colors)
    plt.show()


if __name__ == '__main__':
    for i in range(0, 5):
        show_sample_image()
    show_histogram()
