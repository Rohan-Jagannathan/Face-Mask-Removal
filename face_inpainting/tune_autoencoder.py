import tensorflow as tf
import keras_tuner as kt

from autoencoder import get_model
from preprocess_data import get_data


def build_model(hp):
    model = get_model()
    model.compile(tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')), loss=tf.keras.losses.MSE)

    return model


def tune():
    train_data, test_data = get_data()

    tuner = kt.Hyperband(
        build_model,
        objective='loss',
        max_epochs=50,
        hyperband_iterations=2)

    tuner.search(train_data,
                 epochs=10,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)])

    print(tuner.get_best_hyperparameters()[0].get('learning_rate'))

if __name__ == '__main__':
    tune()
