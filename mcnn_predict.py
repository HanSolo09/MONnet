import cv2
import numpy as np
from gen_dataset import create_multiscale_roi
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from skimage import segmentation
from skimage.measure import regionprops

test_sets = ['top_mosaic_09cm_area26.tif']

input = './data/train/'
output = './data/predict/'
num_segmentaion = 20000

img_w0 = 16
img_h0 = 16
img_w1 = 32
img_h1 = 32
img_w2 = 64
img_h2 = 64


def predict():
    # load the trained convolutional neural network
    print("loading network...")
    model = load_model(input + 'weights.hdf5')
    for i in range(len(test_sets)):
        # load test image
        img = cv2.imread('./data/src/' + test_sets[i])
        row, col, _ = img.shape
        res = np.zeros((row, col, 1), np.uint8)

        # slic superpixels segmentation
        seg = segmentation.slic(img, num_segmentaion)
        seg = seg + 1
        props = regionprops(seg)

        for j in range(len(props)):
            row_j = props[j].coords[0][0]
            col_j = props[j].coords[0][1]
            roi0, roi1, roi2 = create_multiscale_roi(img, seg, props, row_j, col_j)

            roi0 = cv2.resize(roi0, (img_h0, img_w0))
            roi0 = img_to_array(roi0)
            roi0 = np.array(roi0, dtype="float") / 255.0
            roi0 = np.expand_dims(roi0, axis=0)

            roi1 = cv2.resize(roi1, (img_h1, img_w1))
            roi1 = img_to_array(roi1)
            roi1 = np.array(roi1, dtype="float") / 255.0
            roi1 = np.expand_dims(roi1, axis=0)

            roi2 = cv2.resize(roi2, (img_h2, img_w2))
            roi2 = img_to_array(roi2)
            roi2 = np.array(roi2, dtype="float") / 255.0
            roi2 = np.expand_dims(roi2, axis=0)

            prob = model.predict([roi0, roi1, roi2])
            classes = prob.argmax(axis=-1)
            for coord in props[j].coords:
                res[coord[0], coord[1]] = classes[0]

        cv2.imwrite(output + str(i) + '.png', res)


if __name__ == '__main__':
    predict()
