import sys
import numpy as np
import datetime

from gendata import *

sys.path.append('../utils/')
from utils import post_processing
from utils import evaluation

from keras.models import load_model

image_sets = [
    'top_mosaic_09cm_area1.tif',
    'top_mosaic_09cm_area2.tif',
    'top_mosaic_09cm_area3.tif',
    'top_mosaic_09cm_area4.tif',
    'top_mosaic_09cm_area5.tif',
    'top_mosaic_09cm_area6.tif',
    'top_mosaic_09cm_area7.tif',
    'top_mosaic_09cm_area8.tif',
    'top_mosaic_09cm_area10.tif',
    # 'top_mosaic_09cm_area11.tif',
    # 'top_mosaic_09cm_area12.tif',
    # 'top_mosaic_09cm_area13.tif',
    # 'top_mosaic_09cm_area14.tif',
    # 'top_mosaic_09cm_area15.tif',
    # 'top_mosaic_09cm_area16.tif',
    # 'top_mosaic_09cm_area17.tif',
    # 'top_mosaic_09cm_area20.tif',
    # 'top_mosaic_09cm_area21.tif',
    # 'top_mosaic_09cm_area22.tif',
    # 'top_mosaic_09cm_area23.tif',
    # 'top_mosaic_09cm_area24.tif',
    'top_mosaic_09cm_area26.tif'
    # 'top_mosaic_09cm_area27.tif',
    # 'top_mosaic_09cm_area28.tif',
    # 'top_mosaic_09cm_area30.tif',
    # 'top_mosaic_09cm_area31.tif',
    # 'top_mosaic_09cm_area32.tif',
    # 'top_mosaic_09cm_area33.tif',
    # 'top_mosaic_09cm_area34.tif',
    # 'top_mosaic_09cm_area35.tif',
    # 'top_mosaic_09cm_area37.tif',
    # 'top_mosaic_09cm_area38.tif'

    # '2'
    # '3',
    # '4',
    # '7',
    # '8',
    # '9',
    # '10',
    # '11',
    # '13',
    # '14',
    # '15',
    # '16',
    # '17',
    # '18',
    # '19',
    # '20',
    # '21',
    # '22',
    # '23',
    # '24',
    # '25',
    # '26',
    # '27',
    # '28',
    # '29',
    # '32',
    # '33',
    # '34'
]

# imgpath = '/home/ubuntu/Desktop/xiangliu/data/patch2/'
imgpath = './data/src/'
outputpath = './data/predict/'

img_w0 = 24
img_h0 = 24
img_w1 = 48
img_h1 = 48
img_w2 = 72
img_h2 = 72

# 1 means don't use vote
vote_when_predict = 3


def predict_patch_input():
    """
    Predicting based on trained network, single patch input.
    """
    model = load_model('./data/train/weights.hdf5')
    for i in range(len(image_sets)):
        # load test image
        filepath = imgpath + image_sets[i]
        print("predicting " + filepath)
        starttime = datetime.datetime.now()
        rgb_img = cv2.imread(filepath)

        row, col, _ = rgb_img.shape

        res = np.zeros((row, col, 1), np.uint8)
        for r in len(row):
            for c in len(col):
                roi = create_patch_roi(rgb_img, r, c)
                roi = cv2.resize(roi, (img_h0, img_w0))

                # data augmentation
                roi_aug_list = data_augment(roi, vote_when_predict)
                classes_list = []
                for k in range(len(roi_aug_list)):
                    roi_k = np.array(roi_aug_list[k], dtype="float") / 255.0
                    roi_k = np.expand_dims(roi_k, axis=0)
                    prob = model.predict(roi_k)
                    classes_k = prob.argmax(axis=-1)
                    classes_list.append(classes_k[0])

                # vote when predicting
                classes = max(set(classes_list), key=classes_list.count)
                res[r, c] = classes

        cv2.imwrite(outputpath + image_sets[i] + '_pred.png', res)
        print(outputpath + image_sets[i] + '_pred.png is saved!')
        endtime = datetime.datetime.now()
        print('processing time: ')
        print(endtime - starttime)


def predict_single_input():
    """
    Predicting based on trained network, single object input.
    """
    model = load_model('./data/train/weights.hdf5')
    for i in range(len(image_sets)):
        # load test image
        filepath = imgpath + image_sets[i]
        print("predicting " + filepath)
        starttime = datetime.datetime.now()
        rgb_img = cv2.imread(filepath)

        row, col, _ = rgb_img.shape

        # superpixels segmentation
        seg = segmentation.quickshift(rgb_img, kernel_size=3, max_dist=6, ratio=0.5)
        seg = seg + 1
        props = regionprops(seg)

        res = np.zeros((row, col, 1), np.uint8)
        for j in range(len(props)):
            row_j = props[j].coords[0][0]
            col_j = props[j].coords[0][1]
            roi = create_singlescale_roi(rgb_img, seg, props, row_j, col_j)
            roi = cv2.resize(roi, (img_h0, img_w0))

            # data augmentation
            roi_aug_list = data_augment(roi, vote_when_predict)
            classes_list = []
            for k in range(len(roi_aug_list)):
                roi_k = np.array(roi_aug_list[k], dtype="float") / 255.0
                roi_k = np.expand_dims(roi_k, axis=0)
                prob = model.predict(roi_k)
                classes_k = prob.argmax(axis=-1)
                classes_list.append(classes_k[0])

            # vote when predicting
            classes = max(set(classes_list), key=classes_list.count)
            coord_list = props[j].coords
            res[coord_list[:, 0], coord_list[:, 1]] = classes

        cv2.imwrite(outputpath + image_sets[i] + '_pred.png', res)
        print(outputpath + image_sets[i] + '_pred.png is saved!')
        endtime = datetime.datetime.now()
        print('processing time: ')
        print(endtime - starttime)


def predict_multi_input():
    """
    Predicting based on trained network, multi object inputs.
    """
    model = load_model('./data/train/weights.hdf5')
    for i in range(len(image_sets)):
        # load test image
        filepath = imgpath + image_sets[i]
        print("predicting " + filepath)
        starttime = datetime.datetime.now()
        rgb_img = cv2.imread(filepath)

        row, col, _ = rgb_img.shape

        # superpixels segmentation
        seg = segmentation.quickshift(rgb_img, kernel_size=3, max_dist=6, ratio=0.5)
        seg = seg + 1
        props = regionprops(seg)

        res = np.zeros((row, col, 1), np.uint8)
        for j in range(len(props)):
            row_j = props[j].coords[0][0]
            col_j = props[j].coords[0][1]
            roi0, roi1, roi2 = create_multiscale_roi(rgb_img, seg, props, row_j, col_j)
            roi0 = cv2.resize(roi0, (img_h0, img_w0))
            roi1 = cv2.resize(roi1, (img_h1, img_w1))
            roi2 = cv2.resize(roi2, (img_h2, img_w2))
            # data augmentation
            roi0_aug_list = data_augment(roi0, vote_when_predict)
            roi1_aug_list = data_augment(roi1, vote_when_predict)
            roi2_aug_list = data_augment(roi2, vote_when_predict)
            classes_list = []
            for k in range(len(roi0_aug_list)):
                roi0_k = np.array(roi0_aug_list[k], dtype="float") / 255.0
                roi1_k = np.array(roi1_aug_list[k], dtype="float") / 255.0
                roi2_k = np.array(roi2_aug_list[k], dtype="float") / 255.0
                roi0_k = np.expand_dims(roi0_k, axis=0)
                roi1_k = np.expand_dims(roi1_k, axis=0)
                roi2_k = np.expand_dims(roi2_k, axis=0)

                prob = model.predict([roi0_k, roi1_k, roi2_k])
                classes_k = prob.argmax(axis=-1)
                classes_list.append(classes_k[0])

            # vote when predicting
            classes = max(set(classes_list), key=classes_list.count)
            coord_list = props[j].coords
            res[coord_list[:, 0], coord_list[:, 1]] = classes

        cv2.imwrite(outputpath + image_sets[i] + '_pred.png', res)
        print(outputpath + image_sets[i] + '_pred.png is saved!')
        endtime = datetime.datetime.now()
        print('processing time: ')
        print(endtime - starttime)


def do_vote():
    """
    Do voting post processing in the prediction folder recursively.
    """
    for imgfile in image_sets:
        seg_path = outputpath + imgfile + '_seg.png'
        img_path = outputpath + imgfile + '_pred.png'
        seg = cv2.imread(seg_path, -1)
        img = cv2.imread(img_path)

        if seg is None:
            print('Segmentation image not exists!')
            continue

        cv2.imwrite(outputpath + imgfile + '_vote.png', post_processing.vote(img, seg))


def do_crf():
    """
    CRF processing all results in the prediction folder recursively.
    """
    for imgfile in image_sets:
        img_path = imgpath + imgfile
        pred_path = outputpath + imgfile + '_pred.png'
        rgb_img = cv2.imread(img_path)
        pred = cv2.imread(pred_path, -1)
        print("crf " + pred_path)

        cv2.imwrite(outputpath + imgfile + '_crf.png', post_processing.crf(rgb_img, pred, zero_unsure=False))


def do_visualizing(dataset_type):
    """
    Visualizing all results in the prediction folder recursively.
    """
    for imgfile in image_sets:
        pred_path = outputpath + imgfile + '_crf.png'
        pred = cv2.imread(pred_path, -1)
        print("visualizing " + pred_path)

        cv2.imwrite(outputpath + imgfile + '_crf_vis.png', post_processing.draw(pred, dataset_type))


def do_evaluation(csvpath):
    df = pandas.read_csv(csvpath, names=['image', 'filename', 'row', 'col', 'label'], header=0)
    images = df.image.tolist()
    filenames = df.filename.tolist()
    rows = df.row.tolist()
    cols = df.col.tolist()
    labels = df.label.tolist()

    lookuptbl={}
    for i in range(len(images)):
        row_col_label=[rows[i], cols[i],labels[i]]
        if images[i] not in lookuptbl:
            lookuptbl[images[i]]=[]
        else:
            lookuptbl[images[i]].append(row_col_label)

    y_pred = []
    y_true = []
    for imagename in lookuptbl:
        pred_path = outputpath + imagename + '_pred.png'
        pred = cv2.imread(pred_path, -1)

        rows_cols_labels=lookuptbl[imagename]
        for row_col_label in rows_cols_labels:
            pred_i = pred[row_col_label[0], row_col_label[1]]
            y_pred.append(pred_i)
            y_true.append(row_col_label[2])


    evaluation.compute(y_pred=y_pred, y_true=y_true)


if __name__ == '__main__':
    # predict
    # predict_multi_input()
    # predict_single_input()
    # predict_patch_input()

    # post processing
    # do_vote()
    # do_crf()
    do_visualizing(dataset_type='vaihingen')

    # evaluation
    # do_evaluation('./data/train/test_list.csv')

    # todo: IoU evaluation
