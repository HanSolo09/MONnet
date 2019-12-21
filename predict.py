import sys
import datetime
from skimage import io

from gendata import *

sys.path.append('../utils/')
from utils import post_processing
from utils import evaluation

from keras.models import load_model

image_sets = [
    # 'top_mosaic_09cm_area1.tif',
    # 'top_mosaic_09cm_area2.tif',
    # 'top_mosaic_09cm_area3.tif',
    # 'top_mosaic_09cm_area4.tif',
    # 'top_mosaic_09cm_area5.tif',
    # 'top_mosaic_09cm_area6.tif',
    # 'top_mosaic_09cm_area7.tif',
    # 'top_mosaic_09cm_area8.tif',
    # 'top_mosaic_09cm_area10.tif',
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
    # 'top_mosaic_09cm_area26.tif'
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

    # '2.tif',
    # '3.tif',
    # '8.tif',
    # '9.tif',
    # '14.tif',
    # '16.tif',
    '20.tif',
    '22.tif',
    '26.tif',
    '28.tif',
    '33.tif'

    # '2.tif'
    # '3.tif',
    # '4.tif',
    # '7.tif',
    # '8.tif'
    # '9.tif',
    # '10.tif',
    # '11.tif',
    # '13.tif',
    # '14.tif',
    # '15.tif',
    # '16.tif',
    # '17.tif',
    # '18.tif',
    # '19.tif',
    # '20.tif',
    # '21.tif',
    # '22.tif',
    # '23.tif',
    # '24.tif',
    # '25.tif',
    # '26.tif',
    # '27.tif',
    # '28.tif',
    # '29.tif',
    # '32.tif',
    # '33.tif',
    # '34.tif'
]

imgpath = '/home/ubuntu/Desktop/xiangliu/data/patch2/'
# imgpath = './data/src/'
outputpath = './data/predict/'

img_w0 = 24
img_h0 = 24
img_w1 = 48
img_h1 = 48
img_w2 = 72
img_h2 = 72

# 1 means don't use vote
vote_when_predict = 3


def predict_patch_input(input3D=False, batch_size=64):
    """
    Predicting based on trained network, single patch input, i.e. pixel wise.
    :param input3D: if you use SSRN, input3D need to be set as True.
    :param batch_size: predict on batch!  It is recommended to pick a batch size
        that is as large as you can afford without going out of memory (since larger
         batches will usually result in faster evaluation/prediction).
    """
    model = load_model('./data/train/weights.hdf5')
    for i in range(len(image_sets)):
        # load test image
        filepath = imgpath + image_sets[i]
        print("predicting " + filepath)
        starttime = datetime.datetime.now()
        rgb_img = cv2.imread(filepath)

        rows, cols, _ = rgb_img.shape

        res = np.zeros((rows, cols, 1), np.uint8)
        rois = []
        counter = 0
        rows_cols = []
        for r in range(rows):
            for c in range(cols):
                roi = create_patch_roi(rgb_img, r, c)
                roi = cv2.resize(roi, (img_h2, img_w2))

                roi_k = np.array(roi, dtype="float") / 255.0
                if input3D is True:
                    roi_k = np.expand_dims(roi_k, axis=-1)
                rois.append(roi_k)
                rows_cols.append([r, c])
                counter = counter + 1

                # predict on batch
                if (counter is batch_size) or (r is rows - 1 and c is cols - 1):
                    rois = np.array(rois)
                    probs = model.predict(rois, batch_size=batch_size)
                    for j in range(len(probs)):
                        prob = probs[j]
                        row = rows_cols[j][0]
                        col = rows_cols[j][1]
                        res[row, col] = prob.argmax(axis=-1)
                    counter = 0
                    rois = []
                    rows_cols = []

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

        rows, cols, _ = rgb_img.shape

        # superpixels segmentation
        seg_path = imgpath + image_sets[i] + '_seg.npy'
        if path.exists(seg_path):
            seg = np.load(seg_path)
        else:
            seg = segmentation.quickshift(rgb_img, kernel_size=KERNEL_SIZE, max_dist=MAX_DIST, ratio=RATIO)
            np.save(seg_path, seg)
        seg = seg + 1
        props = regionprops(seg)

        res = np.zeros((rows, cols, 1), np.uint8)
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


def predict_multi_input(batch_size=64):
    """
    Predicting based on trained network, multi object inputs.
    :param batch_size: predict on batch!  It is recommended to pick a batch size
        that is as large as you can afford without going out of memory (since larger
         batches will usually result in faster evaluation/prediction).
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
        seg_path = imgpath + image_sets[i] + '_seg.npy'
        if path.exists(seg_path):
            seg = np.load(seg_path)
        else:
            seg = segmentation.quickshift(rgb_img, kernel_size=KERNEL_SIZE, max_dist=MAX_DIST, ratio=RATIO)
            np.save(seg_path, seg)
        seg = seg + 1
        props = regionprops(seg)

        res = np.zeros((row, col, 1), np.uint8)
        roi0s = []
        roi1s = []
        roi2s = []
        coord_list=[]
        counter = 0
        for j in range(len(props)):
            row_j = props[j].coords[0][0]
            col_j = props[j].coords[0][1]

            roi0, roi1, roi2 = create_multiscale_roi(rgb_img, seg, props, row_j, col_j)
            roi0 = cv2.resize(roi0, (img_h0, img_w0))
            roi1 = cv2.resize(roi1, (img_h1, img_w1))
            roi2 = cv2.resize(roi2, (img_h2, img_w2))
            roi0 = np.array(roi0, dtype="float") / 255.0
            roi1 = np.array(roi1, dtype="float") / 255.0
            roi2 = np.array(roi2, dtype="float") / 255.0

            roi0s.append(roi0)
            roi1s.append(roi1)
            roi2s.append(roi2)
            coord_list.append(props[j].coords)
            counter = counter + 1

            # predict on batch
            if (counter is batch_size) or (j is len(props) - 1):
                probs = model.predict([np.array(roi0s), np.array(roi1s), np.array(roi2s)], batch_size=batch_size)
                for k in range(len(probs)):
                    prob = probs[k]
                    coord=coord_list[k]
                    res[coord[:, 0], coord[:, 1]] = prob.argmax(axis=-1)

                roi0s = []
                roi1s = []
                roi2s = []
                coord_list=[]
                counter = 0

        cv2.imwrite(outputpath + image_sets[i] + '_pred.png', res)
        print(outputpath + image_sets[i] + '_pred.png is saved!')
        endtime = datetime.datetime.now()
        print('processing time: ')
        print(endtime - starttime)


def do_vote(segpath, exclude_labels=[]):
    """
    Do voting post processing in the prediction folder recursively.
    :param segpath: greater scale segmentation image
    :param exclude_labels: exclude label list like "cars"
    """
    for imgfile in image_sets:
        seg_path = segpath + imgfile
        pred_path = outputpath + imgfile + '_pred.png'
        seg = io.imread(seg_path)  # should use skimage for io
        img = cv2.imread(pred_path, -1)
        print("voting " + pred_path)

        if seg is None:
            print('Segmentation image not exists!')
            continue

        vote_img = post_processing.vote(img, seg)

        # recover small class like "car"
        for exclude_label in exclude_labels:
            vote_img[img[:, :] == exclude_label] = exclude_label

        cv2.imwrite(outputpath + imgfile + '_pred2.png', vote_img.astype('uint8'))


def do_crf(exclude_labels=[]):
    """
    CRF processing all results in the prediction folder recursively.
    :param exclude_labels: exclude label list like "cars"
    """
    for imgfile in image_sets:
        img_path = imgpath + imgfile
        pred_path = outputpath + imgfile + '_pred2.png'
        rgb_img = cv2.imread(img_path)
        pred = cv2.imread(pred_path, -1)
        print("crf " + pred_path)

        crf_img = post_processing.crf(rgb_img, pred, zero_unsure=False)
        # recover small class like "car"
        for exclude_label in exclude_labels:
            crf_img[pred[:, :] == exclude_label] = exclude_label

        cv2.imwrite(outputpath + imgfile + '_pred2.png', crf_img)


def do_visualizing(dataset_type):
    """
    Visualizing all results in the prediction folder recursively.
    """
    for imgfile in image_sets:
        pred_path = outputpath + imgfile + '_pred2.png'
        pred = cv2.imread(pred_path, -1)
        print("visualizing " + pred_path)

        cv2.imwrite(outputpath + imgfile + '_vis.png', post_processing.draw(pred, dataset_type))


def do_evaluation(csvpath):
    """
    Evaluate test_list, using OA and kappa coefficient.
    :param csvpath: path to test_list.csv
    """
    df = pandas.read_csv(csvpath, names=['image', 'filename', 'row', 'col', 'label'], header=0)
    images = df.image.tolist()
    rows = df.row.tolist()
    cols = df.col.tolist()
    labels = df.label.tolist()

    lookuptbl = {}
    for i in range(len(images)):
        row_col_label = [rows[i], cols[i], labels[i]]
        if images[i] not in lookuptbl:
            lookuptbl[images[i]] = []
        else:
            lookuptbl[images[i]].append(row_col_label)

    y_pred = []
    y_true = []
    for imagename in lookuptbl:
        pred_path = outputpath + imagename + '_pred2.png'
        pred = cv2.imread(pred_path, -1)

        rows_cols_labels = lookuptbl[imagename]
        for row_col_label in rows_cols_labels:
            pred_i = pred[row_col_label[0], row_col_label[1]]
            y_pred.append(pred_i)
            y_true.append(row_col_label[2])

    evaluation.compute(y_pred=y_pred, y_true=y_true)


def do_evaluation_v2(labelpath):
    """
    Evaluate IoU between predicted and ground truth images.
    :param labelpath: ground truth label image
    """
    y_pred = []
    y_true = []
    for imgfile in image_sets:
        pred_path = outputpath + imgfile + '_pred2.png'
        pred = cv2.imread(pred_path, -1)
        temp = np.reshape(pred, (-1, 1))
        temp = temp[:, 0]
        y_pred.extend(temp.tolist())

        gt_path = labelpath + imgfile
        true = cv2.imread(gt_path)
        rows, cols, _ = true.shape
        for r in range(rows):
            for c in range(cols):
                label = bgr2label(true[r, c, :])
                y_true.append(label)

    evaluation.compute_IoU(y_pred=y_pred, y_true=y_true)


if __name__ == '__main__':
    # predict
    # predict_multi_input(batch_size=256)
    # predict_single_input()
    predict_patch_input(input3D=False, batch_size=256)

    # post processing
    do_vote(segpath='./data/optimizing/', exclude_labels=[])
    do_crf(exclude_labels=[])
    # do_visualizing(dataset_type='vaihingen')
    do_visualizing(dataset_type='xiangliu')

    # evaluation
    do_evaluation('./data/train/test_list.csv')

    # mIoU evaluation
    # do_evaluation_v2('./data/label/')
