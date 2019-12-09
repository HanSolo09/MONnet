import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Flatten, MaxPooling2D, Dropout, ZeroPadding2D
from keras.regularizers import l2
from keras.optimizers import Adam


def PixelCNN(shape, n_label):
    """
    Alexnet
    """
    # Initialize model
    model = Sequential()

    # Layer 1
    model.add(Conv2D(3, (11, 11), input_shape=shape,
                     padding='same', kernel_regularizer=l2(0.)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Conv2D(8, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Layer 5
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    model.add(Flatten())
    model.add(Dense(96))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Layer 7
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Layer 8
    model.add(Dense(n_label))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])

    # model.summary()
    # keras.utils.plot_model(model, to_file='PixelCNN.png', show_shapes=True)

    return model
