"""# Model Architecture

Define Double Convolution Downsampling Block
"""


def downsample(inputs, n_filters=64, kernel_dims=3, p_dropout=0.0, pool_size=2):
    # Double convolution
    conv = tf.keras.layers.Conv2D(n_filters, kernel_size=kernel_dims, padding='same', activation='relu',
                                  kernel_initializer='he_normal')(inputs)
    conv = tf.keras.layers.Conv2D(n_filters, kernel_size=kernel_dims, padding='same', activation='relu',
                                  kernel_initializer='he_normal')(conv)

    # Add dropout with specified probability
    if p_dropout > 0.0:
        conv = tf.keras.layers.Dropout(p_dropout)(conv)

    skip = conv

    # Perform Max Pooling with specified pool size
    if pool_size > 0:
        pool = tf.keras.layers.MaxPool2D((pool_size, pool_size))(conv)
    else:
        pool = conv

    return pool, skip


"""Define Double Transpose Convolution Upsampling Block"""


def upsample(prev_inputs, skip_inputs, n_filters=64, kernel_dims=3, stride=2):
    # Transpose convolution
    conv_t1 = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size=(kernel_dims, kernel_dims),
                                              strides=(stride, stride), padding='same')(prev_inputs)

    # Concatenate transpose convolution with skip layer
    concat = tf.keras.layers.concatenate([conv_t1, skip_inputs], axis=3)

    conv_1 = tf.keras.layers.Conv2D(n_filters, kernel_size=kernel_dims, padding='same',
                                    activation='relu', kernel_initializer='he_normal')(concat)

    conv_2 = tf.keras.layers.Conv2D(n_filters, kernel_size=kernel_dims, padding='same',
                                    activation='relu', kernel_initializer='he_normal')(conv_1)

    return conv_2


"""Define UNet"""

def create_unet(input_shape, n_filters, n_classes):
    inputs = tf.keras.layers.Input(input_shape)

    down1 = downsample(inputs, n_filters)
    down2 = downsample(down1[0], n_filters * 2)
    down3 = downsample(down2[0], n_filters * 4)
    down4 = downsample(down3[0], n_filters * 8, pool_size=0)
    up1 = upsample(down4[0], down3[1], n_filters * 4)
    up2 = upsample(up1, down2[1], n_filters * 2)
    up3 = upsample(up2, down1[1], n_filters)

    conv = tf.keras.layers.Conv2D(n_filters, kernel_size=3, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(up3)

    conv2 = tf.keras.layers.Conv2D(n_classes, kernel_size=1, activation='softmax', padding='same')(conv)

    model = tf.keras.Model(inputs=inputs, outputs=conv2)

    return model
