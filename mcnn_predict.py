import cv2
import numpy as np
from gen_dataset import create_multiscale_roi
from keras.models import load_model
from skimage import segmentation
from skimage.measure import regionprops
from sklearn.metrics import confusion_matrix

image_sets = [
    # 'top_mosaic_09cm_area1.tif'
    # 'top_mosaic_09cm_area2.tif',
    # 'top_mosaic_09cm_area3.tif',
    # 'top_mosaic_09cm_area4.tif',
    # 'top_mosaic_09cm_area5.tif',
    # 'top_mosaic_09cm_area6.tif',
    # 'top_mosaic_09cm_area7.tif',
    # 'top_mosaic_09cm_area8.tif',
    # 'top_mosaic_09cm_area10.tif',
    # 'top_mosaic_09cm_area26.tif'

    '2',
    '14',
    '20',
    '3',
    '8',
    '9'
]

imgpath = '/home/ubuntu/Desktop/xiangliu/data/patch2/'
# imgpath='/home/ubuntu/Desktop/gjh2.0/data/src/'
num_segmentaion = 20000

img_w0 = 24
img_h0 = 24
img_w1 = 48
img_h1 = 48
img_w2 = 72
img_h2 = 72


def compute_iou(y_pred, y_true, labels):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels)
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return IoU


def predict():
    # load the trained convolutional neural network
    print("loading network...")
    model = load_model('./data/train/weights.hdf5')
    for i in range(len(image_sets)):
        # load test image

        rgb_img = cv2.imread(imgpath + image_sets[i] + '.tif')
        # rgb_img = rgb_img[4500::,4500::,:]

        # rgb_img = cv2.imread(imgpath + image_sets[i])

        row, col, _ = rgb_img.shape

        # superpixels segmentation
        seg = segmentation.quickshift(rgb_img, kernel_size=3, max_dist=6, ratio=0.5)
        # seg = segmentation.slic(img, num_segmentaion)
        seg = seg + 1
        props = regionprops(seg)

        res = np.zeros((row, col, 1), np.uint8)
        for j in range(len(props)):
            row_j = props[j].coords[0][0]
            col_j = props[j].coords[0][1]
            roi0, roi1, roi2 = create_multiscale_roi(rgb_img, seg, props, row_j, col_j)

            roi0 = cv2.resize(roi0, (img_h0, img_w0))
            roi0 = np.array(roi0, dtype="float") / 255.0
            # roi0 = img_to_array(roi0)
            roi0 = np.expand_dims(roi0, axis=0)

            roi1 = cv2.resize(roi1, (img_h1, img_w1))
            roi1 = np.array(roi1, dtype="float") / 255.0
            # roi1 = img_to_array(roi1)
            roi1 = np.expand_dims(roi1, axis=0)

            roi2 = cv2.resize(roi2, (img_h2, img_w2))
            roi2 = np.array(roi2, dtype="float") / 255.0
            # roi2 = img_to_array(roi2)
            roi2 = np.expand_dims(roi2, axis=0)

            prob = model.predict([roi0, roi1, roi2])

            # todo: 增加投票机制
            classes = prob.argmax(axis=-1)
            for coord in props[j].coords:
                res[coord[0], coord[1]] = classes[0]

        cv2.imwrite(imgpath + image_sets[i] + '_pred.png', res)
        print(imgpath + image_sets[i] + '_pred.png is saved!')


def draw_labels():
    print("drawing labels...")
    for img in image_sets:
        filepath = imgpath + img + '_pred.png'
        img = cv2.imread(filepath, -1)
        row, col = img.shape
        res = np.zeros((row, col, 3), np.uint8)

        for i in range(row):
            for j in range(col):
                pixel = img[i, j].tolist()
                if pixel == 0:
                    res[i, j] = [255, 255, 255]  # Road
                elif pixel == 1:
                    res[i, j] = [0, 0, 255]  # Water
                elif pixel == 2:
                    res[i, j] = [255, 0, 0]  # Buildings
                elif pixel == 3:
                    res[i, j] = [0, 255, 255]  # Cars
                elif pixel == 4:
                    res[i, j] = [0, 255, 0]  # Trees
                elif pixel == 5:
                    res[i, j] = [255, 255, 0]  # Grass
                elif pixel == 6:
                    res[i, j] = [255, 255, 0]  # Grass
                elif pixel == 7:
                    res[i, j] = [255, 255, 0]  # Grass
                elif pixel == 8:
                    res[i, j] = [255, 255, 0]  # Grass


        cv2.imwrite(imgpath + img + '_vis.png', res)


if __name__ == '__main__':
    predict()
    # draw_labels()
