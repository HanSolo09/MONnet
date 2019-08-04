import cv2
import random
import numpy as np
import pandas
from math import *
from skimage import segmentation
from skimage.measure import regionprops

# make sure these folders exist
input='./data/'
output='./data/train/'
image_sets=['top_mosaic_09cm_area26.tif']

sample_margin=200
num_segmentaion=20000

def rotate(img, angle):
    row,col,_=img.shape

    row_new = int(col * fabs(sin(radians(angle))) + row * fabs(cos(radians(angle))))
    col_new = int(row * fabs(sin(radians(angle))) + col * fabs(cos(radians(angle))))

    rotate_matrix = cv2.getRotationMatrix2D((col/2, row/2), angle, 1)

    rotate_matrix[0, 2] += (col_new - col) / 2
    rotate_matrix[1, 2] += (row_new - row) / 2

    img = cv2.warpAffine(img, rotate_matrix, (col_new, row_new), borderValue=(255,255,255))

    return img

def data_augment(roi0, roi1, roi2):
    rand=np.random.random()
    if rand < 0.25:
        roi0 = rotate(roi0, 90)
        roi1 = rotate(roi1, 90)
        roi2 = rotate(roi2, 90)
    elif 0.25 < rand < 0.5:
        roi0 = rotate(roi0, 180)
        roi1 = rotate(roi1, 180)
        roi2 = rotate(roi2, 180)
    elif 0.5 < rand < 0.75:
        roi0 = rotate(roi0, 270)
        roi1 = rotate(roi1, 270)
        roi2 = rotate(roi2, 270)
    elif rand > 0.75:
        # flip by y axis
        roi0 = cv2.flip(roi0, 1)
        roi1 = cv2.flip(roi1, 1)
        roi2 = cv2.flip(roi2, 1)

    return roi0, roi1, roi2

# Vaihingen Dataset
def bgr2label(pixel):
    pixel=pixel.tolist()
    if pixel==[255,255,255]: # Road
        return 0
    elif pixel==[0,0,255]: # Water
        return 1
    elif pixel==[255,0,0]:  # Buildings
        return 2
    elif pixel==[0,255,255]:  # Cars
        return 3
    elif pixel==[0,255,0]:  # Trees
        return 4
    elif pixel==[255,255,0]:  # Grass
        return 5

# create objected-oriented roi according to its scale
def create_multiscale_roi(img, seg, props, row, col):
    seg_id = seg[row, col]-1
    bbox = props[seg_id].bbox
    min_row = bbox[0]
    min_col = bbox[1]
    max_row = bbox[2]
    max_col = bbox[3]

    unique_id = np.unique(seg[min_row:max_row, min_col: max_col]) - 1

    # scale 0
    sub_img_0 = img[min_row:max_row, min_col: max_col, :]

    # scale 1
    for id in unique_id:
        centroid = props[id].centroid
        centroid = np.around(centroid).astype(np.int)

        if centroid[0] < min_row:
            min_row = centroid[0]
        if centroid[0] > max_row:
            max_row = centroid[0]
        if centroid[1] < min_col:
            min_col = centroid[1]
        if centroid[1] > max_col:
            max_col = centroid[1]
    sub_img_1 = img[min_row:max_row, min_col: max_col, :]

    # scale 2
    for id in unique_id:
        bbox = props[id].bbox
        if bbox[0] < min_row:
            min_row = bbox[0]
        if bbox[1] < min_col:
            min_col = bbox[1]
        if bbox[2] > max_row:
            max_row = bbox[2]
        if bbox[3] > max_col:
            max_col = bbox[3]
    sub_img_2 = img[min_row:max_row, min_col: max_col, :]

    return sub_img_0,sub_img_1,sub_img_2

def creat_dataset(num_sample=1000):
    print('creating dataset...')
    count = 0

    for i in range(len(image_sets)):
        # load image and label (both in 3 channels)
        src_img = cv2.imread(input+'src/' + image_sets[i])
        label_img = cv2.imread(input+'label/' + image_sets[i])
        row, col, _ = src_img.shape

        # random sample coordinate
        rows=[]
        cols=[]
        labels=[]
        for j in range(num_sample):
            row_j=random.randint(sample_margin, row - sample_margin)
            col_j=random.randint(sample_margin, col - sample_margin)
            label_j=bgr2label(label_img[row_j,col_j,:])
            rows.append(row_j)
            cols.append(col_j)
            labels.append(label_j)
        df = pandas.DataFrame(data={"row": rows, "col": cols, "label": labels})
        df.to_csv(input+'samples.csv', sep=',', index=False)
        print('random sample coordinate is saved')

        # slic superpixels segmentation
        seg = segmentation.slic(src_img, num_segmentaion)
        seg = seg + 1
        props = regionprops(seg)

        for j in range(num_sample):
            roi0, roi1, roi2 = create_multiscale_roi(src_img, seg, props, rows[j], cols[j])

            # data augmentation
            for k in range(5):
                roi0, roi1, roi2 = data_augment(roi0, roi1, roi2)
                cv2.imwrite(output + '0/' + str(count) + '.tif', roi0)
                cv2.imwrite(output + '1/' + str(count) + '.tif', roi1)
                cv2.imwrite(output + '2/' + str(count) + '.tif', roi2)
                count += 1


if __name__=='__main__':
    creat_dataset()