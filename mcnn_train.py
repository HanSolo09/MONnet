import os
import random

import cv2
import numpy as np
import pandas
from keras import initializers
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, concatenate, Reshape
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from keras.optimizers import Adam

n_label = 6
filepath = './data/train/'
lookup = {}

img_w0 = 16
img_h0 = 16
img_w1 = 32
img_h1 = 32
img_w2 = 64
img_h2 = 64


def load_img(path):
    img = cv2.imread(path)
    img = np.array(img, dtype="float") / 255.0

    return img


def get_train_val(val_rate=0.1):
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

            label = lookup[url]
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


def mcnn():
    input1 = Input(shape=(img_w0, img_h0, 3), name='input1')
    input2 = Input(shape=(img_w1, img_h1, 3), name='input2')
    input3 = Input(shape=(img_w2, img_h2, 3), name='input3')

    # input1
    conv1_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                     kernel_initializer=initializers.truncated_normal(stddev=0.1),
                     bias_initializer=initializers.constant(value=0.1))(input1)

    pool1_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_1)

    conv2_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                     kernel_initializer=initializers.truncated_normal(stddev=0.1),
                     bias_initializer=initializers.constant(value=0.1))(pool1_1)
    pool2_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv2_1)

    # input2
    conv1_2 = Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same',
                     kernel_initializer=initializers.truncated_normal(stddev=0.1),
                     bias_initializer=initializers.constant(value=0.1))(input2)
    pool1_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_2)

    conv2_2 = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                     kernel_initializer=initializers.truncated_normal(stddev=0.1),
                     bias_initializer=initializers.constant(value=0.1))(pool1_2)
    pool2_2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv2_2)

    # input3
    conv1_3 = Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same',
                     kernel_initializer=initializers.truncated_normal(stddev=0.1),
                     bias_initializer=initializers.constant(value=0.1))(input3)
    pool1_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_3)

    conv2_3 = Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same',
                     kernel_initializer=initializers.truncated_normal(stddev=0.1),
                     bias_initializer=initializers.constant(value=0.1))(pool1_3)
    pool2_3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv2_3)

    merged_feature = concatenate([pool2_1, pool2_2, pool2_3], axis=-1)

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

    model = Model(inputs=[input1, input2, input3], outputs=softmax_linear)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
    # model.summary()
    return model


def train():
    EPOCHS = 16
    BS = 64
    model = mcnn()
    modelcheck = ModelCheckpoint(filepath=filepath + 'weights.hdf5', monitor='val_acc', save_best_only=True, mode='max')
    callable = [modelcheck]
    train_set, val_set = get_train_val()
    train_num = len(train_set)
    valid_num = len(val_set)
    print("the number of train data is", train_num)
    print("the number of val data is", valid_num)
    temp = pandas.read_csv(filepath + 'train.csv')
    for i in range(len(temp)):
        lookup[temp.values[i][0]] = temp.values[i][3]

    generateData(BS, train_set)
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


if __name__ == '__main__':
    train()
