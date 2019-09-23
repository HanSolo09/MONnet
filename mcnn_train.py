import os
import random
import cv2
import numpy as np
import pandas
import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import initializers
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, concatenate, Reshape, Input
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

n_label = 6
filepath = './data/train/'
lookup = []

n_channel = 3
img_w0 = 16
img_h0 = 16
img_w1 = 32
img_h1 = 32
img_w2 = 64
img_h2 = 64

LOSS = 'categorical_crossentropy'
VALIDATION_RATE = 0.1


def load_img(path):
    img = cv2.imread(path)
    img = np.array(img, dtype="float") / 255.0

    return img


def get_train_val(val_rate=VALIDATION_RATE):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + '0'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


def generateData(batch_size, data=[]):
    # print('generateData...')
    while True:
        train_data0 = []
        train_data1 = []
        train_data2 = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            roi0 = load_img(filepath + '0/' + url)
            roi0 = cv2.resize(roi0, (img_h0, img_w0))
            roi0 = img_to_array(roi0)
            train_data0.append(roi0)
            roi1 = load_img(filepath + '1/' + url)
            roi1 = cv2.resize(roi1, (img_h1, img_w1))
            roi1 = img_to_array(roi1)
            train_data1.append(roi1)
            roi2 = load_img(filepath + '2/' + url)
            roi2 = cv2.resize(roi2, (img_h2, img_w2))
            roi2 = img_to_array(roi2)
            train_data2.append(roi2)

            label = lookup[int(url.split('.')[0])]
            train_label.append(label)
            if batch % batch_size == 0:
                # print('get enough bacth!')
                train_data0 = np.array(train_data0)
                train_data1 = np.array(train_data1)
                train_data2 = np.array(train_data2)
                train_label = np.array(train_label).flatten()
                train_label = train_label.reshape((batch_size, 1))
                train_label = to_categorical(train_label, num_classes=n_label)
                yield ([train_data0, train_data1, train_data2], train_label)
                train_data0 = []
                train_data1 = []
                train_data2 = []
                train_label = []
                batch = 0


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def multiscale_resnet_v1(inputs, scale, depth=20, num_classes=6):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.

    if scale == 0:

        num_filters = 8
        num_res_blocks = int((depth - 2) / 6)

        x = resnet_layer(inputs=inputs,
                         num_filters=num_filters)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):

                strides = 1
                if stack > 1 and res_block == 0:  # first layer but not first and second stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

    if scale == 1:

        num_filters = 4
        num_res_blocks = int((depth - 2) / 6)

        x = resnet_layer(inputs=inputs,
                         num_filters=num_filters)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first and second stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2
    if scale == 2:
        num_filters = 2
        num_res_blocks = int((depth - 2) / 6)
        x = resnet_layer(inputs=inputs,
                         num_filters=num_filters)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if res_block == 0:  # first layer but not first and second stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

    return x


def mcnn():
    input0 = Input(shape=(img_w0, img_h0, n_channel), name='input0')
    input1 = Input(shape=(img_w1, img_h1, n_channel), name='input1')
    input2 = Input(shape=(img_w2, img_h2, n_channel), name='input2')

    x0 = multiscale_resnet_v1(input0, scale=0)
    x1 = multiscale_resnet_v1(input1, scale=1)
    x2 = multiscale_resnet_v1(input2, scale=2)
    merged_feature = concatenate([x0, x1, x2], axis=-1)
    merged_feature = Reshape((-1,))(merged_feature)
    local3 = Dense(128, activation='relu', use_bias=True, kernel_initializer
    =initializers.truncated_normal(stddev=0.005), bias_initializer
                   =initializers.constant(value=0.1))(merged_feature)

    local4 = Dense(128, activation='relu', use_bias=True, kernel_initializer
    =initializers.truncated_normal(stddev=0.005), bias_initializer
                   =initializers.constant(value=0.1))(local3)

    softmax_linear = Dense(n_label, activation='softmax', use_bias=True, kernel_initializer
    =initializers.truncated_normal(stddev=0.005), bias_initializer
                           =initializers.constant(value=0.1))(local4)

    model = Model(inputs=[input0, input1, input2], outputs=softmax_linear)
    model.compile(optimizer=Adam(lr=0.0001), loss=LOSS,
                  metrics=['binary_crossentropy', 'accuracy'])

    # model.summary()
    # keras.utils.plot_model(model, to_file='model2.png', show_shapes=True)

    return model


def train():
    EPOCHS = 100
    BS = 128
    model = mcnn()
    modelcheck = ModelCheckpoint(filepath=filepath + 'weights.hdf5', monitor='val_acc', save_best_only=True, mode='max')
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callable = [modelcheck,lr_reducer]
    train_set, val_set = get_train_val()
    train_num = len(train_set)
    valid_num = len(val_set)
    print("the number of train data is", train_num)
    print("the number of val data is", valid_num)
    temp = pandas.read_csv(filepath + 'train.csv', names=['label'])
    lookup.extend(temp.label.tolist())
    lookup.pop(0)

    H = model.fit_generator(generator=generateData(BS, train_set), steps_per_epoch=train_num // BS, epochs=EPOCHS,
                            verbose=1,
                            validation_data=generateData(BS, val_set), validation_steps=valid_num // BS,
                            callbacks=callable, max_queue_size=1)


    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on MCNN")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(filepath + "plot.png")

    with open(filepath + 'param.txt', 'a') as f:
        f.write('EPOCHS: ' + str(EPOCHS) + '\n')
        f.write('BS: ' + str(BS) + '\n')
        f.write('LOSS: ' + LOSS + '\n')
        f.write('VALIDATION_RATE: ' + str(VALIDATION_RATE) + '\n')


if __name__ == '__main__':
    train()
