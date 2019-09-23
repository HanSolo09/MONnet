import cv2
import skimage.io
import random
import numpy as np
import pandas
from skimage import segmentation
from skimage.measure import regionprops
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from keras.preprocessing.image import img_to_array

# make sure these folders exist
output = './data/train/'
image_sets = [
    'top_mosaic_09cm_area24.tif',
    'top_mosaic_09cm_area26.tif',
    'top_mosaic_09cm_area27.tif',
    'top_mosaic_09cm_area28.tif',
    'top_mosaic_09cm_area29.tif',
    'top_mosaic_09cm_area30.tif',
    'top_mosaic_09cm_area31.tif',
    'top_mosaic_09cm_area32.tif',
    'top_mosaic_09cm_area33.tif',
    'top_mosaic_09cm_area34.tif'

]

dsm_sets = [
    'dsm_09cm_matching_area24.tif',
    'dsm_09cm_matching_area26.tif',
    'dsm_09cm_matching_area27.tif',
    'dsm_09cm_matching_area28.tif',
    'dsm_09cm_matching_area29.tif',
    'dsm_09cm_matching_area30.tif',
    'dsm_09cm_matching_area31.tif',
    'dsm_09cm_matching_area32.tif',
    'dsm_09cm_matching_area33.tif',
    'dsm_09cm_matching_area34.tif'
]

sample_margin = 200
num_segmentaion = 20000

KERNEL_SIZE = 3
MAX_DIST = 6
RATIO = 0.5

NUM_SAMPLE=5000


datagen = ImageDataGenerator(brightness_range=[0.2,1.0],
                             rotation_range=90,
                             horizontal_flip=True,
                             vertical_flip=True)

# def data_augment(roi0, roi1, roi2,rand):
#     rand = np.random.random()
#     if rand == 0:
#         roi0 = roi0
#         roi1 = roi1
#         roi2 = roi2
#     elif rand ==1:
#         roi0 = np.rot90(roi0, 2)
#         roi1 = np.rot90(roi1, 2)
#         roi2 = np.rot90(roi2, 2)
#     elif rand==2:
#         roi0 = np.rot90(roi0, 3)
#         roi1 = np.rot90(roi1, 3)
#         roi2 = np.rot90(roi2, 3)
#     elif rand ==3:
#         # flip by y axis
#         roi0 = cv2.flip(roi0, 1)
#         roi1 = cv2.flip(roi1, 1)
#         roi2 = cv2.flip(roi2, 1)
#
#     return roi0, roi1, roi2
def data_augment(img):
    img = expand_dims(img,0)
    it = datagen.flow(img, batch_size=9)
    img_aug_list = []
    for i in range(9):
        batch = it.next()
        img_aug = batch[0].astype('uint8')
        img_aug_list.append(img_aug)
    return img_aug_list





# Vaihingen Dataset
def bgr2label(pixel):
    pixel = pixel.tolist()
    if pixel == [255, 255, 255]:  # Road
        return 0
    elif pixel == [0, 0, 255]:  # Water
        return 1
    elif pixel == [255, 0, 0]:  # Buildings
        return 2
    elif pixel == [0, 255, 255]:  # Cars
        return 3
    elif pixel == [0, 255, 0]:  # Trees
        return 4
    elif pixel == [255, 255, 0]:  # Grass
        return 5


# create objected-oriented roi according to its scale
def create_multiscale_roi(img, seg, props, row, col):
    seg_id = seg[row, col] - 1
    bbox = props[seg_id].bbox
    min_row = bbox[0]
    min_col = bbox[1]
    max_row = bbox[2]
    max_col = bbox[3]

    unique_id = np.unique(seg[min_row:max_row, min_col: max_col]) - 1

    # scale 0
    sub_img_0 = img[min_row:max_row, min_col: max_col, :]

    # scale 1x
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

    return sub_img_0, sub_img_1, sub_img_2


def creat_dataset(num_sample=NUM_SAMPLE):
    print('creating dataset...')
    count = 0
    filenames = []
    rows = []
    cols = []
    labels = []
    for i in range(len(image_sets)):
        # load image and label (both in 3 channels)
        print('creating dataset from image ' + str(i))
        rgb_img = cv2.imread('./data/src/' + image_sets[i])
        row, col, _ = rgb_img.shape
        # dsm_img = cv2.imread('./data/src/' + dsm_sets[i], -1)
        # dsm_img = dsm_img.reshape(row, col, 1)
        # dsm_img = np.interp(dsm_img, (dsm_img.min(), dsm_img.max()), (0, 1))
        # src_img = np.concatenate((rgb_img, dsm_img), axis=-1)
        src_img=rgb_img
        label_img = cv2.imread('./data/label/' + image_sets[i])

        # slic superpixels segmentation
        seg = segmentation.quickshift(rgb_img, kernel_size=KERNEL_SIZE, max_dist=MAX_DIST, ratio=RATIO)
        # seg = segmentation.slic(src_img, num_segmentaion)
        seg = seg + 1
        props = regionprops(seg)

        # random sample
        for j in range(num_sample):
            row_j = random.randint(sample_margin, row - sample_margin)
            col_j = random.randint(sample_margin, col - sample_margin)
            label_j = bgr2label(label_img[row_j, col_j, :])

            roi0, roi1, roi2 = create_multiscale_roi(src_img, seg, props, row_j, col_j)
            # save origin patch
            cv2.imwrite(output + '0/' + str(count).zfill(7) + '.png', roi0.astype('uint8'))
            cv2.imwrite(output + '1/' + str(count).zfill(7) + '.png', roi1.astype('uint8'))
            cv2.imwrite(output + '2/' + str(count).zfill(7) + '.png', roi2.astype('uint8'))
            filenames.append(str(count).zfill(7) + '.png')
            rows.append(row_j)
            cols.append(col_j)
            labels.append(label_j)
            count += 1
            # data augmentation
            roi0_aug_list = data_augment(roi0)
            roi1_aug_list = data_augment(roi1)
            roi2_aug_list = data_augment(roi2)
            for k in range(len(roi0_aug_list)):
                cv2.imwrite(output + '0/' + str(count).zfill(7) + '.png', roi0_aug_list[k])
                cv2.imwrite(output + '1/' + str(count).zfill(7) + '.png', roi1_aug_list[k])
                cv2.imwrite(output + '2/' + str(count).zfill(7) + '.png', roi2_aug_list[k])
                filenames.append(str(count).zfill(7) + '.png')
                rows.append(row_j)
                cols.append(col_j)
                labels.append(label_j)
                count += 1

            # for k in range(5):
            #     roi0, roi1, roi2 = data_augment(roi0, roi1, roi2)
            #     cv2.imwrite(output + '0/' + str(count).zfill(7) + '.tif', roi0)
            #     cv2.imwrite(output + '1/' + str(count).zfill(7) + '.tif', roi1)
            #     cv2.imwrite(output + '2/' + str(count).zfill(7) + '.tif', roi2)
            #     filenames.append(str(count).zfill(7) + '.tif')
            #     rows.append(row_j)
            #     cols.append(col_j)
            #     labels.append(label_j)
            #     count += 1

    df = pandas.DataFrame(data={"filename": filenames, "row": rows, "col": cols, "label": labels})
    df.to_csv(output + 'train.csv', sep=',', index=False)

    with open(output + 'param.txt', 'w') as f:
        f.write('KERNEL_SIZE: ' + str(KERNEL_SIZE) + '\n')
        f.write('MAX_DIST: ' + str(MAX_DIST) + '\n')
        f.write('RATIO: ' + str(RATIO) + '\n')
        f.write('NUM_SAMPLE: ' + str(num_sample) + '\n')
        f.write('NUM_SEGMENTATION: ' + str(num_segmentaion) + '\n')
        f.write('IMAGE_SETS: \n')
        for image in image_sets:
            f.write(image + '\n')
        f.close()

    print('All saved')


if __name__ == '__main__':
    creat_dataset()
