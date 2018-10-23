from keras import Input, Model
from keras.initializers import glorot_uniform
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Add, BatchNormalization, Activation, \
    AveragePooling2D, Flatten, Dense, ZeroPadding2D, LeakyReLU, LocallyConnected2D, Dropout, Concatenate

from trainer.constants import IM_SHAPE, IM_SHAPE_BBOX

# UNet is NOT my work; copied directly from https://github.com/zhixuhao/unet.
def UNet(input_size=IM_SHAPE):
    inputs = Input(input_size)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def YoloReduced(input_shape=IM_SHAPE_BBOX):
    inputs = Input(input_shape)

    X = conv_block(inputs, 64, (7, 7), (2, 2))
    X = MaxPooling2D((2, 2), (2, 2))(X)
    X = conv_block(X, 192, (3, 3), (1, 1))
    X = MaxPooling2D((2, 2), (2, 2))(X)
    X = conv_block(X, 128, (1, 1), (1, 1))
    X = conv_block(X, 256, (3, 3), (1, 1))
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = MaxPooling2D((2, 2), (2, 2))(X)
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = conv_block(X, 512, (1, 1), (1, 1))
    X = conv_block(X, 1024, (3, 3), (1, 1))
    X = conv_block(X, 512, (1, 1), (1, 1))
    X = conv_block(X, 1024, (3, 3), (1, 1))
    X = conv_block(X, 1024, (3, 3), (1, 1))
    X = conv_block(X, 1024, (3, 3), (2, 2))
    X = conv_block(X, 1024, (3, 3), (1, 1))
    X = conv_block(X, 1024, (3, 3), (1, 1))
    X = local_block(X, 256, (3, 3), (1, 1))
    X = Flatten()(X)
    X = Dense(1000)(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dropout(0.3)(X)
    outputs = Dense(5, activation='linear')(X)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def HigherResCNN(input_shape=IM_SHAPE_BBOX):
    inputs = Input(input_shape)

    X = conv_block(inputs, 32, (3, 3), (1, 1))
    X = conv_block(X, 64, (3, 3), (2, 2))  # Downsample
    X_shortcut = X
    X = conv_block(X, 32, (1, 1), (1, 1))
    X = conv_block(X, 64, (3, 3), (1, 1))
    X = Add()([X_shortcut, X])
    X = conv_block(X, 128, (3, 3), (2, 2))  # Downsample
    X_shortcut = X
    X = conv_block(X, 64, (1, 1), (1, 1))
    X = conv_block(X, 128, (3, 3), (1, 1))
    X = Add()([X_shortcut, X])
    X_shortcut = X
    X = conv_block(X, 64, (1, 1), (1, 1))
    X = conv_block(X, 128, (3, 3), (1, 1))
    X = Add()([X_shortcut, X])
    X = conv_block(X, 256, (3, 3), (2, 2))  # Downsample
    X = shortcut_block(X, 128)
    X = shortcut_block(X, 128)
    X = shortcut_block(X, 128)
    X = shortcut_block(X, 128)
    X = shortcut_block(X, 128)
    X = shortcut_block(X, 128)
    X = shortcut_block(X, 128)
    X = shortcut_block(X, 128)
    X = conv_block(X, 512, (3, 3), (2, 2))  # Downsample
    X = shortcut_block(X, 256)
    X = shortcut_block(X, 256)
    X = shortcut_block(X, 256)
    X = shortcut_block(X, 256)
    X = shortcut_block(X, 256)
    X = shortcut_block(X, 256)
    X = conv_block(X, 1024, (3, 3), (2, 2))  # Downsample
    X = shortcut_block(X, 512)
    X = shortcut_block(X, 512)
    X = shortcut_block(X, 512)
    X = shortcut_block(X, 512)
    X = Flatten()(X)
    X = Dense(1024)(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dropout(0.4)(X)
    outputs = Dense(5, activation='linear')(X)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def shortcut_block(X, filters):
    X_shortcut = X
    X = conv_block(X, filters, (1, 1), (1, 1))
    X = conv_block(X, filters * 2, (3, 3), (1, 1))
    return Add()([X_shortcut, X])


def CustomCNN(input_shape=IM_SHAPE_BBOX, output_shape=5):
    inputs = Input(input_shape)

    X = conv_block(inputs, 64, (7, 7), (2, 2))
    X = MaxPooling2D((2, 2), (2, 2))(X)
    X = conv_block(X, 192, (3, 3), (1, 1))
    X = MaxPooling2D((2, 2), (2, 2))(X)
    X = conv_block(X, 128, (1, 1), (1, 1))
    X = conv_block(X, 256, (3, 3), (1, 1))
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = MaxPooling2D((2, 2), (2, 2))(X)
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = local_block(X, 256, (3, 3), (1, 1))
    X = Flatten()(X)
    X = Dense(500)(X)
    outputs = Dense(output_shape, activation='linear')(X)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def CustomCNN1(input_shape=IM_SHAPE_BBOX, output_shape=5):
    inputs = Input(input_shape)
    X = conv_block(inputs, 64, (7, 7), (2, 2))
    X = MaxPooling2D((2, 2), (2, 2))(X)
    X = conv_block(X, 192, (3, 3), (1, 1))
    X = MaxPooling2D((2, 2), (2, 2))(X)
    X = conv_block(X, 128, (1, 1), (1, 1))
    X = conv_block(X, 256, (3, 3), (1, 1))
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = MaxPooling2D((2, 2), (2, 2))(X)
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = conv_block(X, 256, (1, 1), (1, 1))
    X = conv_block(X, 512, (3, 3), (1, 1))
    X = conv_block(X, 1024, (3, 3), (1, 1))
    X = conv_block(X, 1024, (3, 3), (1, 1))
    X = local_block(X, 256, (3, 3), (1, 1))
    X = Flatten()(X)
    X = Dense(1000)(X)
    X = Dropout(0.35)(X)
    X = LeakyReLU(alpha=0.1)(X)
    outputs = Dense(output_shape, activation='linear')(X)
    # outputs = Dropout(0.35)(X)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# def BinaryModel(input_shape=IM_SHAPE_BBOX, categories=2):
#     inputs = Input(input_shape)
#
#     X = BatchNormalization()(inputs)
#     X = conv_block(X, 64, (7, 7), (2, 2))
#     X = MaxPooling2D((2, 2), (2, 2))(X)
#     X = conv_block(X, 192, (3, 3), (1, 1))
#     X = MaxPooling2D((2, 2), (2, 2))(X)
#     X = conv_block(X, 128, (1, 1), (1, 1))
#     X = conv_block(X, 256, (3, 3), (1, 1))
#     X = MaxPooling2D((2, 2), (2, 2))(X)
#     X = conv_block(X, 256, (1, 1), (1, 1))
#     X = local_block(X, 256, (3, 3), (1, 1))
#     X = Flatten()(X)
#     X = Dense(1000)(X)
#     X = Dropout(0.4)(X)
#     outputs = Dense(output_shape, activation='linear')(X)
#
#     model = Model(inputs=inputs, outputs=outputs)
#     return model


def conv_block(X, filters, kernel_size, strides):
    X = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.1)(X)
    return X


def local_block(X, filters, kernel_size, strides):
    X = ZeroPadding2D((1, 1))(X)
    X = LocallyConnected2D(filters, kernel_size, strides=strides, use_bias=False)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.1)(X)
    return X
