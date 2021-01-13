import cv2
import os
import argparse
import numpy as np
import pandas
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from model.MONet import MONet
from model.MONetv2 import MONetv2
from model.PixelCNN import PixelCNN
from model.SSRN import SSRN
from model.SingleCNN import SingleCNN
from model.Unet import Unet

unet_gt_dir = '/home/ubuntu/data/mcnn_data/vaihingen_final/unet_label/'

n_channel = 3
img_w0 = 24
img_h0 = 24
img_w1 = 48
img_h1 = 48
img_w2 = 72
img_h2 = 72


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='MONet training')
    parser.add_argument('--model_type',
                        default='MONet', type=str,
                        help='Model type')
    parser.add_argument('--n_label',
                        default=6, type=int,
                        help='Number of classes in the training data')
    parser.add_argument('--train_data_dir',
                        default='/home/guojinhui/data/MONet_data/train_data', type=str,
                        help='Input training data directory')
    parser.add_argument('--output_dir',
                        default='/home/guojinhui/data/MONet_data/training/20210111', type=str,
                        help='Directory used to save model weights')
    parser.add_argument('--epochs',
                        default=64, type=int,
                        help='Training epochs')
    parser.add_argument('--batch_size',
                        default=128, type=int,
                        help='Batch size')
    parser.add_argument('--cuda_visible_devices',
                        default='0', type=str,
                        help='GPU used when training and prediction')

    global args
    args = parser.parse_args(argv)
    for arg in vars(args):
        print(arg, getattr(args, arg))

    if not os.path.exists(args.output_dir):
        print(args.output_dir + ' not exists, making new directory')
        os.makedirs(args.output_dir)


def load_mean_img(path):
    img = cv2.imread(path)
    img = np.array(img, dtype="float") / 255.0

    return img


def generate_data_unet_input(batch_size, images):
    while True:
        train_data = []
        train_label = []
        batch = 0

        for i in (range(len(images))):
            url = str(images[i])
            batch += 1
            roi = load_mean_img(os.path.join(args.train_data_dir, '2/', url))
            roi = cv2.resize(roi, (64, 64))
            roi = img_to_array(roi)
            train_data.append(roi)

            gt = cv2.imread(os.path.join(unet_gt_dir, url), -1)
            gt = cv2.resize(gt, (64, 64))
            gt = np.array(gt).flatten()
            gt = to_categorical(gt, num_classes=args.n_label)
            gt = gt.reshape((64, 64, args.n_label))

            gt = img_to_array(gt)
            train_label.append(gt)

            # if get enough bacth
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)

                train_data = []
                train_label = []
                batch = 0


def generate_data_single_input(batch_size, images, labels, input3D=False):
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(images))):
            url = str(images[i])
            batch += 1
            roi0 = load_mean_img(os.path.join(args.train_data_dir, '2/', url))
            roi0 = img_to_array(roi0)
            train_data.append(roi0)
            train_label.append(labels[i])

            # if get enough bacth
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                if input3D is True:
                    train_data = np.expand_dims(train_data, axis=-1)  # expand to 3D input
                train_label = np.array(train_label).flatten()
                train_label = train_label.reshape((batch_size, 1))
                train_label = to_categorical(train_label, num_classes=args.n_label)
                yield (train_data, train_label)

                train_data = []
                train_label = []
                batch = 0


def generate_data_multiple_input(batch_size, images, labels):
    """
    Generate multiple inputs for training.
    :param batch_size: batch size
    :param images: image filepath lists
    :param labels: label lists
    :return: Multiple inputs and corresponding labels
    """

    while True:
        train_data0 = []
        train_data1 = []
        train_data2 = []
        train_label = []
        batch = 0
        for i in (range(len(images))):
            url = str(images[i])
            batch += 1
            roi0 = load_mean_img(os.path.join(args.train_data_dir, '0/', url))
            roi0 = img_to_array(roi0)
            train_data0.append(roi0)
            roi1 = load_mean_img(os.path.join(args.train_data_dir, '1/', url))
            roi1 = img_to_array(roi1)
            train_data1.append(roi1)
            roi2 = load_mean_img(os.path.join(args.train_data_dir, '2/', url))
            roi2 = img_to_array(roi2)
            train_data2.append(roi2)

            train_label.append(labels[i])

            # get enough batch
            if batch % batch_size == 0:
                train_data0 = np.array(train_data0)
                train_data1 = np.array(train_data1)
                train_data2 = np.array(train_data2)
                train_label = np.array(train_label).flatten()
                train_label = train_label.reshape((batch_size, 1))
                train_label = to_categorical(train_label, num_classes=args.n_label)
                yield ([train_data0, train_data1, train_data2], train_label)

                train_data0 = []
                train_data1 = []
                train_data2 = []
                train_label = []
                batch = 0


def train(model_type):
    """
    Start training the model.
    :param model_type: Model type. Mush be one of 'MONet', 'PixelCNN', 'SSRN' or 'SingleCNN'
    """

    # some callbacks
    modelcheck = ModelCheckpoint(filepath=os.path.join(args.output_dir, model_type + '_weights.hdf5'), monitor='val_accuracy', save_best_only=True, mode='max')
    # format="weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    # modelcheck = ModelCheckpoint(filepath=filepath + format, monitor='val_acc', save_best_only=False, mode='max',period=1)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    callable = [modelcheck, lr_reducer]

    # load train and test set
    df = pandas.read_csv(os.path.join(args.train_data_dir, 'train_list.csv'), names=['image', 'filename', 'row', 'col', 'label'], header=0)
    train_filename = df.filename.tolist()
    train_label = df.label.tolist()
    df2 = pandas.read_csv(os.path.join(args.train_data_dir, 'test_list.csv'), names=['image', 'filename', 'row', 'col', 'label'], header=0)
    test_filename = df2.filename.tolist()
    test_label = df2.label.tolist()
    print("the number of train data is", len(train_filename))
    print("the number of test data is", len(test_filename))

    # training
    if model_type == 'MONet':
        model = MONet((img_h0, img_w0, n_channel), (img_h1, img_w1, n_channel), (img_h2, img_w2, n_channel), args.n_label)
        H = model.fit_generator(generator=generate_data_multiple_input(args.batch_size, train_filename, train_label),
                                steps_per_epoch=len(train_filename) // args.batch_size,
                                epochs=args.epochs,
                                verbose=1,
                                validation_data=generate_data_multiple_input(args.batch_size, test_filename, test_label),
                                validation_steps=len(test_filename) // args.batch_size,
                                callbacks=callable, max_queue_size=1)
    elif model_type == 'MONetv2':
        model = MONetv2((img_h0, img_w0, n_channel), (img_h1, img_w1, n_channel), (img_h2, img_w2, n_channel), args.n_label)
        H = model.fit_generator(generator=generate_data_multiple_input(args.batch_size, train_filename, train_label),
                                steps_per_epoch=len(train_filename) // args.batch_size,
                                epochs=args.epochs,
                                verbose=1,
                                validation_data=generate_data_multiple_input(args.batch_size, test_filename, test_label),
                                validation_steps=len(test_filename) // args.batch_size,
                                callbacks=callable, max_queue_size=1)
    elif model_type == 'PixelCNN':
        model = PixelCNN(shape=(img_h2, img_w2, n_channel), n_label=args.n_label)
        H = model.fit_generator(generator=generate_data_single_input(args.batch_size, train_filename, train_label),
                                steps_per_epoch=len(train_filename) // args.batch_size,
                                epochs=args.epochs,
                                verbose=1,
                                validation_data=generate_data_single_input(args.batch_size, test_filename, test_label),
                                validation_steps=len(test_filename) // args.batch_size,
                                callbacks=callable, max_queue_size=1)
    elif model_type == 'SSRN':
        model = SSRN(shape=(1, img_h2, img_w2, n_channel), n_label=args.n_label)
        H = model.fit_generator(generator=generate_data_single_input(args.batch_size, train_filename, train_label, input3D=True),
                                steps_per_epoch=len(train_filename) // args.batch_size,
                                epochs=args.epochs,
                                verbose=1,
                                validation_data=generate_data_single_input(args.batch_size, test_filename, test_label, input3D=True),
                                validation_steps=len(test_filename) // args.batch_size,
                                callbacks=callable, max_queue_size=1)
    elif model_type == 'SingleCNN':
        model = SingleCNN(shape=(img_h2, img_w2, n_channel), n_label=args.n_label, scale=2)
        H = model.fit_generator(generator=generate_data_single_input(args.batch_size, train_filename, train_label),
                                steps_per_epoch=len(train_filename) // args.batch_size,
                                epochs=args.epochs,
                                verbose=1,
                                validation_data=generate_data_single_input(args.batch_size, test_filename, test_label),
                                validation_steps=len(test_filename) // args.batch_size,
                                callbacks=callable, max_queue_size=1)
    elif model_type == 'Unet':
        model = Unet(shape=(64, 64, n_channel), n_label=args.n_label)
        train_filename = train_filename[::10]
        H = model.fit_generator(generator=generate_data_unet_input(args.batch_size, train_filename),
                                steps_per_epoch=len(train_filename) // args.batch_size,
                                epochs=args.epochs,
                                verbose=1,
                                validation_data=generate_data_unet_input(args.batch_size, test_filename),
                                validation_steps=len(test_filename) // args.batch_size,
                                callbacks=callable, max_queue_size=1)
    else:
        raise ValueError('Model type is wrong')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, args.epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, args.epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, args.epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, args.epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on " + model_type)
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(args.output_dir, model_type + "_plot.png"))


if __name__ == '__main__':
    parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    train(model_type=args.model_type)
