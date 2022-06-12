import matplotlib.pyplot as plt
from random import randint

from autoencoder import get_model
from preprocess_data import *


def view_results(indices):
    masked, segmaps, unmasked, inverse_segmaps, removed_masks = get_data()
    x, y = get_xy()

    model = get_model()
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.000605186189165695),
                  loss=tf.keras.losses.MSE)
    model.load_weights(CHECKPOINT_PATH + '/inpainting_weights_tuned_2500_0024.h5')

    whole_data = tf.data.Dataset.from_tensors((x, y))
    outputs = model.predict(whole_data, batch_size=16)

    for index in indices:
        figure = plt.figure(figsize=(12, 12))

        removed = figure.add_subplot(231)
        plt.imshow(x[index])

        expected = figure.add_subplot(232)
        plt.imshow(y[index])

        reconstructed = figure.add_subplot(233)
        plt.imshow(outputs[index])

        inpainted = figure.add_subplot(234)
        inpainted_image = (outputs[index]) * (np.repeat(np.expand_dims(segmaps[index], axis=-1), 3, axis=-1))
        plt.imshow(inpainted_image)

        merged = figure.add_subplot(235)
        plt.imshow(inpainted_image + x[index])

        removed.title.set_text('Removed Mask')
        expected.title.set_text('Original Image')
        reconstructed.title.set_text('Reconstructed Image')
        inpainted.title.set_text('Inpainted Section')
        merged.title.set_text('Merged Image')

        plt.show()


if __name__ == '__main__':
    idxs = []
    for i in range(0, 5):
        idxs.append(randint(test_start, DATASET_SIZE))
    view_results(idxs)

