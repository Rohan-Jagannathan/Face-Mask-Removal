import tensorflow as tf

from constants import *


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((height, width, n_channels)),

        # 104x88x3 -> 52x44x32
        tf.keras.layers.Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu',
                               kernel_initializer='he_normal'),
        tf.keras.layers.MaxPool2D((2, 2)),

        # 52x44x32 -> 26x22x64
        tf.keras.layers.Conv2D(filters=n_filters * 2, kernel_size=3, padding='same', activation='relu',
                               kernel_initializer='he_normal'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.BatchNormalization(),

        # 26x22x64 -> 13x11x128
        tf.keras.layers.Conv2D(filters=n_filters * 4, kernel_size=3, padding='same', activation='relu',
                               kernel_initializer='he_normal'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.BatchNormalization(),

        # 13x11x128 -> 13x11x256
        tf.keras.layers.Conv2D(filters=n_filters * 8, kernel_size=3, padding='same', activation='relu',
                               kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),

        # 13x11x256 -> 13x11x512
        tf.keras.layers.Conv2D(filters=n_filters * 16, kernel_size=3, padding='same', activation='relu',
                               kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),

        # 13x11x512 -> 26x22x256
        tf.keras.layers.Conv2DTranspose(filters=n_filters * 8, kernel_size=(3, 3), strides=(2, 2), padding='same'),

        # 26x22x256 -> 52x44x128
        tf.keras.layers.Conv2DTranspose(filters=n_filters * 4, kernel_size=(3, 3), strides=(2, 2), padding='same'),

        # 52x44x128 -> 104x88x64
        tf.keras.layers.Conv2DTranspose(filters=n_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same'),

        # 104x88x64 -> 104x88x32
        tf.keras.layers.Conv2DTranspose(filters=n_filters * 1, kernel_size=(3, 3), strides=(1, 1), padding='same'),

        # 108x88x32 -> 108x88x3
        tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', kernel_initializer='he_normal')
    ])

    return model
