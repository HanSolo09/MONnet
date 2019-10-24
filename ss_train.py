import cv2
import numpy as np
import pandas
import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.initializers import glorot_uniform
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, concatenate, Flatten, Input,Reshape
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Reshape
from keras import backend as K

n_label = 6
filepath = './data/train/'
lookup = []

n_channel = 3
img_w0 = 24
img_h0 = 24
img_w1 = 48
img_h1 = 48
img_w2 = 72
img_h2 = 72

LOSS = 'categorical_crossentropy'


def load_img(path):
    img = cv2.imread(path)
    img = np.array(img, dtype="float") / 255.0

    return img


def get_train_val():
    temp = pandas.read_csv(filepath + 'train_list.csv', names=['train_list'])
    train_set = temp.train_list.tolist()
    train_set.pop(0)
    temp2 = pandas.read_csv(filepath + 'validation_list.csv', names=['validation_list'])
    val_set = temp2.validation_list.tolist()
    val_set.pop(0)

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
            # roi0 = cv2.resize(roi0, (img_h0, img_w0))
            roi0 = img_to_array(roi0)
            train_data0.append(roi0)
            roi1 = load_img(filepath + '1/' + url)
            # roi1 = cv2.resize(roi1, (img_h1, img_w1))
            roi1 = img_to_array(roi1)
            train_data1.append(roi1)
            roi2 = load_img(filepath + '2/' + url)
            # roi2 = cv2.resize(roi2, (img_h2, img_w2))
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
                  kernel_initializer=glorot_uniform(seed=0),
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


def res_block(inputs,
              num_filters):
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters)
    x = resnet_layer(inputs=x,
                     num_filters=num_filters,
                     activation=None)
    x = keras.layers.add([inputs, x])
    x = Activation('relu')(x)

    return x


def conv_block(inputs,
               num_filters,
               strides=2):
    x = resnet_layer(inputs=inputs,
                     strides=strides,
                     num_filters=num_filters)
    y = resnet_layer(inputs=x,
                     num_filters=num_filters,
                     activation=None)
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters,
                     kernel_size=1,
                     strides=strides,
                     activation=None,
                     batch_normalization=False)
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    return x


def multiscale_resnet(inputs, scale):
    if scale == 0:
        inputs = resnet_layer(inputs=inputs,
                              num_filters=16)

        # stage 0
        x = conv_block(inputs, 16)
        x = res_block(x, 16)
        x = res_block(x, 16)

    elif scale == 1:
        inputs = resnet_layer(inputs=inputs,
                              num_filters=8)

        # stage 0
        x = conv_block(inputs, 8)
        x = res_block(x, 8)
        x = res_block(x, 8)

        # stage 1
        x = conv_block(x, 16)
        x = res_block(x, 16)
        x = res_block(x, 16)

    elif scale == 2:
        inputs = resnet_layer(inputs=inputs,
                              num_filters=4)

        # stage 0
        x = res_block(inputs, 4)
        x = res_block(x, 4)
        x = res_block(x, 4)

        # stage 1
        x = conv_block(x, 8)
        x = res_block(x, 8)
        x = res_block(x, 8)

        # stage 2
        x = conv_block(x, 16, 3)
        x = res_block(x, 16)
        x = res_block(x, 16)

    return x

def ss():
    input0 = Input(shape=(img_w0, img_h0, n_channel), name='input0')
    input1 = Input(shape=(img_w1, img_h1, n_channel), name='input1')
    input2 = Input(shape=(img_w2, img_h2, n_channel), name='input2')

    x0 = multiscale_resnet(input0, scale=0)
    x1 = multiscale_resnet(input1, scale=1)
    x2 = multiscale_resnet(input2, scale=2)
    merged_feature = concatenate([x0, x1, x2], axis=-1)
    _,w,h,c = K.int_shape(x0)

    merged_feature = Reshape((3,w,h,c))(merged_feature)
    print(merged_feature.shape)
    first_ConvLSTM = ConvLSTM2D(filters=10, kernel_size=(3, 3)
    				   ,kernel_initializer='random_uniform'
                       , padding='same', return_sequences=True)(merged_feature)
    second_ConvLSTM = ConvLSTM2D(filters=10, kernel_size=(3, 3)
                        , data_format='channels_last'
                        , padding='same', return_sequences=True)(first_ConvLSTM)
    last_ConvLSTM = ConvLSTM2D(filters=5, kernel_size=(3, 3)
                        , data_format='channels_last'
                        , stateful = False
                        , kernel_initializer='random_uniform'
                        , padding='same', return_sequences=False)(second_ConvLSTM)

    flatten = Flatten()(last_ConvLSTM)
    softmax_linear = Dense(n_label, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(flatten)

    model = Model(inputs=[input0, input1, input2], outputs=softmax_linear)
    model.compile(optimizer=Adam(lr=0.001), loss=LOSS,
                  metrics=['binary_crossentropy', 'accuracy'])

    # model.summary()
    # keras.utils.plot_model(model, to_file='model3.png', show_shapes=True)

    return model


def train():
    EPOCHS = 64
    BS = 128
    model = ss()
    modelcheck = ModelCheckpoint(filepath=filepath + 'weights.hdf5', monitor='val_acc', save_best_only=True, mode='max')
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callable = [modelcheck, lr_reducer]
    temp = pandas.read_csv(filepath + 'train.csv', names=['label'])
    lookup.extend(temp.label.tolist())
    lookup.pop(0)

    train_set, val_set = get_train_val()
    train_num = len(train_set)
    valid_num = len(val_set)
    print("the number of train data is", train_num)
    print("the number of val data is", valid_num)

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


if __name__ == '__main__':
    train()
