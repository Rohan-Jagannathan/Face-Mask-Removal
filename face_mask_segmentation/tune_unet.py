import keras_tuner as kt

from unet import create_unet
from initialize_data import *


def build_model(hp):
    model = create_unet(input_shape=(height, width, image_channels), n_filters=n_filters, n_classes=n_classes)
    model.compile(tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False))
    return model


def tune():
    train_data, valid_data, test_data = get_data_loaders()

    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=50,
        hyperband_iterations=2)

    tuner.search(train_data,
                 validation_data=valid_data,
                 epochs=10,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])

    print(tuner.get_best_hyperparameters()[0].get('learning_rate'))


if __name__ == '__main__':
    tune()