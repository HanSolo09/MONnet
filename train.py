import cv2
import numpy as np
import pandas
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from model.MCNN import MCNN
from model.PixelCNN import PixelCNN
from model.SSRN import SSRN
from model.SingleCNN import SingleCNN

imgpath = './data/train/'

n_label = 6
n_channel = 3
img_w0 = 24
img_h0 = 24
img_w1 = 48
img_h1 = 48
img_w2 = 72
img_h2 = 72

EPOCHS = 100
BS = 128


def load_mean_img(path):
    img = cv2.imread(path)
    img = np.array(img, dtype="float") / 255.0

    return img


def generate_data_single_input(batch_size, images, labels):
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(images))):
            url = str(images[i])
            batch += 1
            roi0 = load_mean_img(imgpath + '0/' + url)
            roi0 = img_to_array(roi0)
            train_data.append(roi0)
            train_label.append(labels[i])

            # if get enough bacth
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label).flatten()
                train_label = train_label.reshape((batch_size, 1))
                train_label = to_categorical(train_label, num_classes=n_label)
                yield (train_data, train_label)

                train_data = []
                train_label = []
                batch = 0


def generate_data_multiple_input(batch_size, images, labels):
    while True:
        train_data0 = []
        train_data1 = []
        train_data2 = []
        train_label = []
        batch = 0
        for i in (range(len(images))):
            url = str(images[i])
            batch += 1
            roi0 = load_mean_img(imgpath + '0/' + url)
            roi0 = img_to_array(roi0)
            train_data0.append(roi0)
            roi1 = load_mean_img(imgpath + '1/' + url)
            roi1 = img_to_array(roi1)
            train_data1.append(roi1)
            roi2 = load_mean_img(imgpath + '2/' + url)
            roi2 = img_to_array(roi2)
            train_data2.append(roi2)

            train_label.append(labels[i])

            # if get enough bacth
            if batch % batch_size == 0:
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


def train(model_type):
    # some callbacks
    modelcheck = ModelCheckpoint(filepath=imgpath + 'weights.hdf5', monitor='val_acc', save_best_only=True, mode='max')
    # format="weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    # modelcheck = ModelCheckpoint(filepath=filepath + format, monitor='val_acc', save_best_only=False, mode='max',period=1)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    callable = [modelcheck, lr_reducer]

    # load train and test set
    df = pandas.read_csv(imgpath + 'train_list.csv', names=['filename', 'label'], header=0)
    train_filename = df.filename.tolist()
    train_label = df.label.tolist()
    df2 = pandas.read_csv(imgpath + 'test_list.csv', names=['image', 'filename', 'row', 'col', 'label'], header=0)
    test_filename = df2.filename.tolist()
    test_label = df2.label.tolist()
    print("the number of train data is", len(train_filename))
    print("the number of test data is", len(test_filename))

    # training
    if model_type is 'MCNN':
        model = MCNN((img_h0, img_w0, n_channel), (img_h1, img_w1, n_channel), (img_h2, img_w2, n_channel), n_label)
        H = model.fit_generator(generator=generate_data_multiple_input(BS, train_filename, train_label),
                                steps_per_epoch=len(train_filename) // BS,
                                epochs=EPOCHS,
                                verbose=1,
                                validation_data=generate_data_multiple_input(BS, test_filename, test_label),
                                validation_steps=len(test_filename) // BS,
                                callbacks=callable, max_queue_size=1)
    elif model_type is 'PixelCNN':
        model = PixelCNN(shape=(img_h0, img_w0, n_channel), n_label=n_label)
        H = model.fit_generator(generator=generate_data_single_input(BS, train_filename, train_label),
                                steps_per_epoch=len(train_filename) // BS,
                                epochs=EPOCHS,
                                verbose=1,
                                validation_data=generate_data_multiple_input(BS, test_filename, test_label),
                                validation_steps=len(test_filename) // BS,
                                callbacks=callable, max_queue_size=1)
    elif model_type is 'SSRN':
        model = SSRN(shape=(1, img_h0, img_w0, n_channel), n_label=n_label)
        H = model.fit_generator(generator=generate_data_single_input(BS, train_filename, train_label),
                                steps_per_epoch=len(train_filename) // BS,
                                epochs=EPOCHS,
                                verbose=1,
                                validation_data=generate_data_multiple_input(BS, test_filename, test_label),
                                validation_steps=len(test_filename) // BS,
                                callbacks=callable, max_queue_size=1)
    elif model_type is 'SingleCNN':
        model = SingleCNN(shape=(img_h0, img_w0, n_channel), n_label=n_label)
        H = model.fit_generator(generator=generate_data_single_input(BS, train_filename, train_label),
                                steps_per_epoch=len(train_filename) // BS,
                                epochs=EPOCHS,
                                verbose=1,
                                validation_data=generate_data_multiple_input(BS, test_filename, test_label),
                                validation_steps=len(test_filename) // BS,
                                callbacks=callable, max_queue_size=1)
    else:
        print('Model type is wrong')
        return

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
    plt.savefig(imgpath + "plot.png")


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train(model_type='MCNN')
