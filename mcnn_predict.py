import cv2
import numpy as np
from gen_dataset import create_multiscale_roi
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from skimage import segmentation
from skimage.measure import regionprops
from sklearn.metrics import confusion_matrix
import pandas

LABELS = [0,1,2,3,4,5]
test_sets = [

'top_mosaic_09cm_area37.tif'
# 'top_mosaic_09cm_area2.tif',
# 'top_mosaic_09cm_area3.tif',
# 'top_mosaic_09cm_area4.tif',
# 'top_mosaic_09cm_area5.tif',
# 'top_mosaic_09cm_area6.tif'
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
# 'top_mosaic_09cm_area22.tif',
# 'top_mosaic_09cm_area23.tif'
]

dsm_sets = [
# 'dsm_09cm_matching_area1.tif'
# 'dsm_09cm_matching_area2.tif'
# 'dsm_09cm_matching_area3.tif',
# 'dsm_09cm_matching_area4.tif',
# 'dsm_09cm_matching_area5.tif',
# 'dsm_09cm_matching_area6.tif',
# 'dsm_09cm_matching_area7.tif',
# 'dsm_09cm_matching_area8.tif',
# 'dsm_09cm_matching_area10.tif',
# 'dsm_09cm_matching_area11.tif',
# 'dsm_09cm_matching_area12.tif',
# 'dsm_09cm_matching_area13.tif',
# 'dsm_09cm_matching_area14.tif',
# 'dsm_09cm_matching_area15.tif',
# 'dsm_09cm_matching_area16.tif',
# 'dsm_09cm_matching_area17.tif',
# 'dsm_09cm_matching_area20.tif',
# 'dsm_09cm_matching_area22.tif',
# 'dsm_09cm_matching_area23.tif'
]

ground_truth_set = [
'top_mosaic_09cm_area26.tif'
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
# 'top_mosaic_09cm_area22.tif',
# 'top_mosaic_09cm_area23.tif'
]



input = './data/train/'
output = './data/predict/'
num_segmentaion = 20000

img_w0 = 16
img_h0 = 16
img_w1 = 32
img_h1 = 32
img_w2 = 64
img_h2 = 64

def compute_iou(y_pred, y_true,labels):
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
     #return np.mean(IoU)




def predict():
    # load the trained convolutional neural network
    print("loading network...")
    model = load_model(input + 'weights.hdf5')
    mIoU = []
    for i in range(len(test_sets)):
        # load test image
        rgb_img = cv2.imread('./data/src/test/' + test_sets[i])

        row, col, _ = rgb_img.shape
        # dsm_img = cv2.imread('./data/src/test/' + dsm_sets[i], -1)
        # dsm_img = dsm_img.reshape(row, col, 1)
        # dsm_img = np.interp(dsm_img, (dsm_img.min(), dsm_img.max()), (0, 1))
        # src_img = np.concatenate((rgb_img, dsm_img), axis=-1)
        src_img=rgb_img

        # slic superpixels segmentation
        seg = segmentation.quickshift(rgb_img, kernel_size=3, max_dist=6, ratio=0.5)
        # seg = segmentation.slic(img, num_segmentaion)
        seg = seg + 1
        props = regionprops(seg)

        res = np.zeros((row, col, 1), np.uint8)
        for j in range(len(props)):
            row_j = props[j].coords[0][0]
            col_j = props[j].coords[0][1]
            roi0, roi1, roi2 = create_multiscale_roi(src_img, seg, props, row_j, col_j)

            roi0 = cv2.resize(roi0, (img_h0, img_w0))
            roi0 = np.array(roi0, dtype="float") / 255.0
            #roi0 = img_to_array(roi0)
            roi0 = np.expand_dims(roi0, axis=0)

            roi1 = cv2.resize(roi1, (img_h1, img_w1))
            roi1 = np.array(roi1, dtype="float") / 255.0
            #roi1 = img_to_array(roi1)
            roi1 = np.expand_dims(roi1, axis=0)

            roi2 = cv2.resize(roi2, (img_h2, img_w2))
            roi2 = np.array(roi2, dtype="float") / 255.0
            #roi2 = img_to_array(roi2)
            roi2 = np.expand_dims(roi2, axis=0)

            prob = model.predict([roi0, roi1, roi2])
            classes = prob.argmax(axis=-1)
            for coord in props[j].coords:
                res[coord[0], coord[1]] = classes[0]


        cv2.imwrite(output + test_sets[i].split('.')[0]+ '_pred.png', res)
        print(test_sets[i].split('.')[0]+ '_pred.png is saved!')

        # # compute iou
        # miou = compute_iou(res,ground_truth_set[i],LABELS)
        # mIoU.append(miou)
    # df = pandas.DataFrame(data={"filename": test_sets, "mIoU": mIoU})
    # df.to_csv(output + 'mIoU.csv', sep=',', index=False)




if __name__ == '__main__':
    predict()
