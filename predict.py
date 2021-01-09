import sys
import datetime
from skimage import io
from keras.models import load_model

from gen_data import *

sys.path.append('../utils/')
from utils import post_processing
from utils import evaluation

image_names = [
    'top_mosaic_09cm_area1.tif',
    'top_mosaic_09cm_area2.tif',
    'top_mosaic_09cm_area3.tif',
    'top_mosaic_09cm_area4.tif',
    'top_mosaic_09cm_area5.tif',
    'top_mosaic_09cm_area6.tif',
    'top_mosaic_09cm_area7.tif',
    'top_mosaic_09cm_area8.tif',
    'top_mosaic_09cm_area10.tif',
    'top_mosaic_09cm_area26.tif'

    # '2.tif',
    # '3.tif',
    # '8.tif',
    # '9.tif',
    # '14.tif',
    # '16.tif',
    # '20.tif',
    # '22.tif',
    # '26.tif',
    # '28.tif',
    # '33.tif'
]

input_dir = '/home/irsgis/data/MONet_data/image'
output_dir = '/home/irsgis/data/MONet_data/training/20210109'

img_w0 = 24
img_h0 = 24
img_w1 = 48
img_h1 = 48
img_w2 = 72
img_h2 = 72


def predict_unet_input(modelpath):
    """
    Predicting images based on trained network weights, unet input, i.e., in a sliding window manner.
    :param modelpath: file path to model weights
    """
    model = load_model(modelpath)
    for i in range(len(image_names)):
        # load test image
        filepath = os.path.join(input_dir, image_names[i])
        rgb_img = cv2.imread(filepath)
        print("predicting " + filepath)
        starttime = datetime.datetime.now()

        rows, cols, _ = rgb_img.shape
        res = np.zeros((rows, cols), np.uint8)

        stride = 32  # overlay 50%
        for r in np.arange(0, rows, stride):
            for c in np.arange(0, cols, stride):
                roi = DatasetGenerator.create_patch_roi_v2(rgb_img, r, c, 64)
                rows_roi, cols_roi, _ = roi.shape
                roi = cv2.resize(roi, (64, 64))
                roi = np.array(roi, dtype="float") / 255.0
                roi = np.expand_dims(roi, axis=0)

                prob = model.predict(np.array(roi))
                pred = prob.argmax(axis=-1)
                pred = pred.reshape((64, 64)).astype(np.uint8)
                pred = cv2.resize(pred, (cols_roi, rows_roi))

                res[r:min(r + 64, rows - 1), c:min(c + 64, cols - 1)] = pred[:, :]

        cv2.imwrite(os.path.join(output_dir, image_names[i] + '_pred.png'), res)
        print(image_names[i] + '_pred.png is saved!')
        endtime = datetime.datetime.now()
        print('processing time: ')
        print(endtime - starttime)


def predict_patch_input(modelpath, input3D=False, batch_size=64):
    """
    Predicting images based on trained network weights, single patch input, i.e. pixel wise.
    :param modelpath: file path to model weights
    :param input3D: if you use SSRN, input3D need to be set as True.
    :param batch_size: predict on batch!  It is recommended to pick a batch size
        that is as large as you can afford without going out of memory (since larger
         batches will usually result in faster evaluation/prediction).
    """
    model = load_model(modelpath)
    for i in range(len(image_names)):
        # load test image
        filepath = os.path.join(input_dir, image_names[i])
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
                roi = DatasetGenerator.create_patch_roi(rgb_img, r, c, img_w2)
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

        cv2.imwrite(os.path.join(output_dir, image_names[i] + '_pred.png'), res)
        print(image_names[i] + '_pred.png is saved!')
        endtime = datetime.datetime.now()
        print('processing time: ')
        print(endtime - starttime)


def predict_single_input(modelpath, batch_size=64):
    """
    Predicting images based on trained network weights, single object input.
    :param modelpath: file path to model weights
    :param batch_size: predict on batch!  It is recommended to pick a batch size
        that is as large as you can afford without going out of memory (since larger
         batches will usually result in faster evaluation/prediction).
    """
    model = load_model(modelpath)
    for i in range(len(image_names)):
        # load test image
        filepath = os.path.join(input_dir, image_names[i])
        print("predicting " + filepath)
        starttime = datetime.datetime.now()
        rgb_img = cv2.imread(filepath)

        rows, cols, _ = rgb_img.shape

        seg = DatasetGenerator.segmentation(image_path=filepath)
        seg = seg + 1
        props = regionprops(seg)

        res = np.zeros((rows, cols, 1), np.uint8)
        rois = []
        coord_list = []
        counter = 0
        for j in range(len(props)):
            row_j = props[j].coords[0][0]
            col_j = props[j].coords[0][1]

            # todo: change the single input scale according to your network
            roi = DatasetGenerator.create_singlescale_roi(rgb_img, seg, props, row_j, col_j, scale=0)
            roi = cv2.resize(roi, (img_h0, img_w0))

            roi = np.array(roi, dtype="float") / 255.0

            rois.append(roi)
            coord_list.append(props[j].coords)
            counter = counter + 1

            # predict on batch
            if (counter is batch_size) or (j is len(props) - 1):
                probs = model.predict(np.array(rois), batch_size=batch_size)
                for k in range(len(probs)):
                    prob = probs[k]
                    coord = coord_list[k]
                    res[coord[:, 0], coord[:, 1]] = prob.argmax(axis=-1)

                rois = []
                coord_list = []
                counter = 0

        cv2.imwrite(os.path.join(output_dir, image_names[i] + '_pred.png'), res)
        print(image_names[i] + '_pred.png is saved!')
        endtime = datetime.datetime.now()
        print('processing time: ')
        print(endtime - starttime)


def predict_multi_input(modelpath, batch_size=64):
    """
    Predicting images based on trained network weights, multi object inputs.
    :param modelpath: file path to model weights
    :param batch_size: predict on batch!  It is recommended to pick a batch size
        that is as large as you can afford without going out of memory (since larger
         batches will usually result in faster evaluation/prediction).
    """
    model = load_model(modelpath)
    for i in range(len(image_names)):
        # load test image
        filepath = os.path.join(input_dir, image_names[i])
        print("predicting " + filepath)
        starttime = datetime.datetime.now()
        rgb_img = cv2.imread(filepath)

        row, col, _ = rgb_img.shape

        seg = DatasetGenerator.segmentation(image_path=filepath)
        seg = seg + 1
        props = regionprops(seg)

        res = np.zeros((row, col, 1), np.uint8)
        roi0s = []
        roi1s = []
        roi2s = []
        coord_list = []
        counter = 0
        for j in range(len(props)):
            row_j = props[j].coords[0][0]
            col_j = props[j].coords[0][1]

            roi0, roi1, roi2 = DatasetGenerator.create_multiscale_roi(rgb_img, seg, props, row_j, col_j)
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
                    coord = coord_list[k]
                    res[coord[:, 0], coord[:, 1]] = prob.argmax(axis=-1)

                roi0s = []
                roi1s = []
                roi2s = []
                coord_list = []
                counter = 0

        cv2.imwrite(os.path.join(output_dir, image_names[i] + '_pred.png'), res)
        print(image_names[i] + '_pred.png is saved!')
        endtime = datetime.datetime.now()
        print('processing time: ')
        print(endtime - starttime)


def do_vote(seg_dir, exclude_labels=[]):
    """
    Do post-processing (voting) of prediction results.
    :param seg_dir: multiresolution segmentation (MRS) images
    :param exclude_labels: categories excluded during post-processing.
           Since post-processing may cause some small objects like cars being filtered out.
    """
    for imgfile in image_names:
        seg_path = os.path.join(seg_dir, imgfile)
        pred_path = os.path.join(output_dir, imgfile + '_pred.png')
        seg = io.imread(seg_path)  # should use skimage for io
        img = cv2.imread(pred_path, -1)
        print("voting " + pred_path)

        if seg is None:
            print('Segmentation image not exists!')
            continue

        vote_img = post_processing.vote(img, seg)

        # recover small objects
        for exclude_label in exclude_labels:
            vote_img[vote_img[:, :] == exclude_label] = 0
            vote_img[img[:, :] == exclude_label] = exclude_label

        cv2.imwrite(os.path.join(output_dir, imgfile + '_pred2.png'), vote_img.astype('uint8'))


def do_crf(exclude_labels=[], gt_prob=.9, sxy_gaussian=3, sxy_bilateral=60, srgb_bilateral=10):
    """
    CRF processing all results in the prediction folder recursively.
    :param exclude_labels: categories excluded during post-processing.
           Since post-processing may cause some small objects like cars being filtered out.
    """
    for imgfile in image_names:
        img_path = os.path.join(input_dir, imgfile)
        pred_path = os.path.join(output_dir, imgfile + '_pred2.png')
        rgb_img = cv2.imread(img_path)
        pred = cv2.imread(pred_path, -1)
        print("crf " + pred_path)

        crf_img = post_processing.crf(rgb_img, pred, gt_prob=gt_prob, sxy_gaussian=sxy_gaussian, sxy_bilateral=sxy_bilateral, srgb_bilateral=srgb_bilateral)

        # recover small objects
        for exclude_label in exclude_labels:
            crf_img[crf_img[:, :] == exclude_label] = 0
            crf_img[pred[:, :] == exclude_label] = exclude_label

        cv2.imwrite(os.path.join(output_dir, imgfile + '_pred2.png'), crf_img)


def do_visualizing(dataset_type):
    """
    Visualizing all results in the prediction folder recursively.
    :param dataset_type: dataset type. Should be either 'vaihingen' or 'xiangliu'
    """
    for imgfile in image_names:
        pred_path = os.path.join(output_dir, imgfile + '_pred2.png')
        pred = cv2.imread(pred_path, -1)
        print("visualizing " + pred_path)

        cv2.imwrite(os.path.join(output_dir, imgfile + '_vis.png'), post_processing.draw(pred, dataset_type))


def do_evaluation(csvpath, ignore_zero=False):
    """
    Evaluate test samples, using OA and kappa coefficient.
    :param csvpath: path to test_list.csv
    :param ignore_zero: whether ignore zero or not.
        If zero means background, ignore_zero should be set as True (xiangliu dataset).
        Otherwise, ignore_zero should be set as false (vaihingen dataset).
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
        pred_path = os.path.join(output_dir, imagename + '_pred.png')
        pred = cv2.imread(pred_path, -1)

        rows_cols_labels = lookuptbl[imagename]
        for row_col_label in rows_cols_labels:
            pred_i = pred[row_col_label[0], row_col_label[1]]

            if ignore_zero is True:  # skip zero value when evaluation
                if pred_i == 0 or row_col_label[2] == 0:
                    continue

            y_pred.append(pred_i)
            y_true.append(row_col_label[2])

    oa = evaluation.compute(y_pred=y_pred, y_true=y_true)
    return oa


def do_evaluation_v2(labelpath):
    """
    Evaluate IoU between predicted and ground truth images.
    :param labelpath: ground truth label image
    """
    y_pred = []
    y_true = []
    for imgfile in image_names:
        pred_path = os.path.join(output_dir, imgfile + '_pred2.png')
        pred = cv2.imread(pred_path, -1)
        temp = np.reshape(pred, (-1, 1))
        temp = temp[:, 0]
        y_pred.extend(temp.tolist())

        gt_path = labelpath + imgfile
        true = cv2.imread(gt_path)
        rows, cols, _ = true.shape
        for r in range(rows):
            for c in range(cols):
                label = DatasetGenerator.bgr2label(true[r, c, :], dataset_type='vaihingen')
                y_true.append(label)

    evaluation.compute_IoU(y_pred=y_pred, y_true=y_true)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # predict
    predict_multi_input(modelpath='/home/irsgis/data/MONet_data/vaihingen_final/vaihingen_results/mcnn/weights.hdf5', batch_size=256)
    # predict_single_input(modelpath='./data/vaihingen_final/singleCNN_level0/weights.hdf5',batch_size=256)
    # predict_patch_input(modelpath='./data/vaihingen_final/pixelCNN/weights.hdf5',input3D=False, batch_size=256)
    # predict_unet_input(modelpath='/home/ubuntu/data/mcnn_data/vaihingen_final/vaihingen_results/unet/weights.hdf5')

    # post processing
    do_vote(seg_dir='/home/irsgis/data/MONet_data/optimizing', exclude_labels=[3])
    do_crf(exclude_labels=[3])

    do_visualizing(dataset_type='vaihingen')

    # evaluation
    do_evaluation('/home/irsgis/data/MONet_data/train_data/test_list.csv', ignore_zero=False)

    # mIoU evaluation
    # do_evaluation_v2('./data/label/')
