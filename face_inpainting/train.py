import tensorflow as tf
import matplotlib.pyplot as plt

from constants import *
from autoencoder import get_model
from preprocess_data import get_datasets


def train():
    train_data, test_data = get_datasets()

    model = get_model()
    model.summary()

    model.compile(tf.keras.optimizers.Adam(learning_rate=0.000605186189165695),
                  loss=tf.keras.losses.MSE)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH + '/inpainting_weights_unet_2500.h5', monitor='loss', save_best_only=True)
    history = model.fit(train_data, batch_size=16, epochs=2500, verbose=1, shuffle=True, callbacks=[checkpoint])

    plt.plot(history.history['loss'], label='Training Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # Overwrites saved model with the one just trained
    # model.save(MODEL_PATH)

