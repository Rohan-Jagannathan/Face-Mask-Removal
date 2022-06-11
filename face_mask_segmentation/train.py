import matplotlib.pyplot as plt
import visualkeras
from keras.callbacks import ModelCheckpoint, EarlyStopping

from unet import create_unet
from initialize_data import *


def train_model():
    train_data, valid_data, test_data = get_data_loaders()

    model = create_unet(input_shape=(height, width, image_channels),
                        n_filters=n_filters, n_classes=n_classes)

    model.summary()

    model.compile(tf.keras.optimizers.Adam(learning_rate=0.00011183105292845106),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    tf.keras.backend.clear_session()
    checkpoint = ModelCheckpoint(CHECKPOINT_PATH + 'segmentation_norm_80.h5', monitor='val_loss', save_best_only=True)
    history = model.fit(train_data, batch_size=16, epochs=80, verbose=1, shuffle=True, validation_data=valid_data,
                        callbacks=[checkpoint])

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # Overwrites saved model with newly trained one
    # model.save(MODEL_PATH)


if __name__ == '__main__':
    train_model()
