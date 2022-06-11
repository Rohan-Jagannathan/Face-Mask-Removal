import matplotlib.pyplot as plt
import cv2 as cv

from initialize_data import *


def predict():
    train_data, valid_data, test_data = get_data_loaders()
    model = tf.keras.models.load_model(MODEL_PATH)

    outputs = model.predict(test_data, batch_size=16)
    outputs = np.array(outputs).argmax(3)

    return outputs


def view_sample_result(outputs, index):
    inputs, segmaps = get_data()

    post_processed = outputs[index].astype(np.uint8)

    # # remove noise in segmentation map
    kernel2 = np.ones((5, 5), np.uint8)
    post_processed = cv.morphologyEx(post_processed, cv.MORPH_OPEN, kernel2)

    figure = plt.figure(figsize=(15, 15))

    expected = figure.add_subplot(131)
    plt.imshow(segmaps[valid_end + index], 'Greys_r')

    predicted = figure.add_subplot(132)
    plt.imshow(outputs[index], 'Greys_r')

    post = figure.add_subplot(133)
    plt.imshow(post_processed, 'Greys_r')

    # original.title.set_text('Original Image')
    expected.title.set_text('True Segmentation')
    predicted.title.set_text('Predicted Segmentation')
    post.title.set_text('Postprocessed Segmentation')

    plt.show()


if __name__ == '__main__':
    out = predict()
    for i in range(10, 15):
        view_sample_result(out, i)
