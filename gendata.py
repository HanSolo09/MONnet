import cv2
import random
import pandas
from os import path
from skimage import segmentation
from skimage.measure import regionprops
from numpy import expand_dims
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

# make sure these folders exist
output = './data/train/'
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
    # 'top_mosaic_09cm_area26.tif'

    '2.tif',
    '3.tif',
    '8.tif',
    '9.tif',
    '14.tif',
    '16.tif',
    '20.tif',
    '22.tif',
    '26.tif',
    '28.tif',
    '33.tif'
]

KERNEL_SIZE = 3
MAX_DIST = 6
RATIO = 0.5

img_w0 = 24
img_h0 = 24
img_w1 = 48
img_h1 = 48
img_w2 = 72
img_h2 = 72

VALIDATION_RATE = 0.2
NUM_AUGMENT = 10
seed = 1
datagen = ImageDataGenerator(brightness_range=[0.3, 1.2],
                             rotation_range=40,
                             horizontal_flip=True,
                             vertical_flip=True)


def data_augment(img, number):
    """
    Data augment function.
    :param img: origin image
    :param number: number of return image list
    :return: image list, include origin image at the first place.
    """
    img_aug_list = []
    img_aug_list.append(img)
    img = expand_dims(img, 0)
    it = datagen.flow(img, batch_size=number - 1, seed=seed)
    for i in range(number - 1):
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


def create_patch_roi(img, row, col):
    """
    Create patch-based roi, i.e. pixel-wise.
    Single scale.
    :param img: original image
    :param row: row index
    :param col: col index
    :return: roi image
    """
    rows, cols, _ = img.shape
    margin = int(img_w2 / 2)
    sub_img = img[max(row - margin, 0):min(row + margin, rows - 1),
              max(col - margin, 0): min(col + margin, cols - 1), :]
    return sub_img


def create_singlescale_roi(img, seg, props, row, col):
    """
    Create objected-oriented roi, single scale.

    :param img: original image
    :param seg: segmentation image
    :param props: regionprops object
    :param row: sample row index
    :param col: sample col index
    :return: roi image
    """
    seg_id = seg[row, col] - 1

    bbox = props[seg_id].bbox
    min_row = bbox[0]
    min_col = bbox[1]
    max_row = bbox[2]
    max_col = bbox[3]

    rows, cols, _ = img.shape

    # scale 0
    sub_img_0 = img[min_row:max_row, min_col: max_col, :]

    # scale 1
    margin1 = 15
    sub_img_1 = img[max(min_row - margin1, 0):min(max_row + margin1, rows - 1),
                max(min_col - margin1, 0): min(max_col + margin1, cols - 1), :]

    # scale 2
    margin2 = 30
    sub_img_2 = img[max(min_row - margin2, 0):min(max_row + margin2, rows - 1),
                max(min_col - margin2, 0): min(max_col + margin2, cols - 1), :]

    return sub_img_0


def create_multiscale_roi(img, seg, props, row, col):
    """
    Create objected-oriented roi, multi scale.

    :param img: original image
    :param seg: segmentation image
    :param props: regionprops object
    :param row: sample row index
    :param col: sample col index
    :return: 3 scale roi images
    """
    seg_id = seg[row, col] - 1

    bbox = props[seg_id].bbox
    min_row = bbox[0]
    min_col = bbox[1]
    max_row = bbox[2]
    max_col = bbox[3]

    row, col, _ = img.shape

    # scale 0
    sub_img_0 = img[min_row:max_row, min_col: max_col, :]

    # scale 1
    margin1 = 15
    sub_img_1 = img[max(min_row - margin1, 0):min(max_row + margin1, row - 1),
                max(min_col - margin1, 0): min(max_col + margin1, col - 1), :]

    # scale 2
    margin2 = 30
    sub_img_2 = img[max(min_row - margin2, 0):min(max_row + margin2, row - 1),
                max(min_col - margin2, 0): min(max_col + margin2, col - 1), :]

    return sub_img_0, sub_img_1, sub_img_2


def creat_csv_from_gt(imgpath, labelpath, num_sample=5000):
    """
    Create dataset from random samples, use when you already have a ground truth.
    Note that some sample points may fall into the same object, this will be tackled in  creat_dataset_from_csv function.
    Save all.csv and dataset in train folder.

    :param imgpath: source image path
    :param labelpath: ground truth label image path
    :param num_sample: number of samples in each image
    """
    print('Creating csv...')

    for i in range(len(image_sets)):
        # load image and label (both in 3 channels)
        img_path = imgpath + image_sets[i]
        label_path = labelpath + image_sets[i]
        rgb_img = cv2.imread(img_path)
        label_img = cv2.imread(label_path)
        print('Creating csv from ' + img_path)

        row_list = []
        col_list = []
        label_list = []

        # random sample
        sample_margin = 20
        rows, cols, _ = rgb_img.shape
        for j in range(num_sample):
            row_j = random.randint(sample_margin, rows - sample_margin)
            col_j = random.randint(sample_margin, cols - sample_margin)
            label_j = bgr2label(label_img[row_j, col_j, :])
            row_list.append(row_j)
            col_list.append(col_j)
            label_list.append(label_j)

        df = pandas.DataFrame(data={'label': label_list, 'row': row_list, 'col': col_list})
        df.to_csv(output + image_sets[i] + '.csv', sep=',', index=False)

    print('Create csv done! You can create dataset from csv now.\n')


def creat_dataset_from_csv(imgpath, csvpath):
    """
    Create dataset from manual sample point file (.csv format)
    Use when you don't have a ground truth, this will save all.csv and dataset in train folder.

    :param imgpath: source image path
    :param csvpath: sample csv path
        csv format example:
        label,row,col
        0,2355,590
        0,1489,177
        2,946,571
    """
    print('Creating dataset from csv...')

    count = 0
    images = []
    filenames = []
    rows = []
    cols = []
    labels = []
    is_original = []
    for i in range(len(image_sets)):
        rgb_path = imgpath + image_sets[i]
        rgb_img = cv2.imread(rgb_path)
        print('Creating dataset from ' + rgb_path)

        # sample from file
        temp = pandas.read_csv(csvpath + image_sets[i] + '.csv', names=['label', 'row', 'col'], header=0)
        temp = temp.dropna()
        rowlist = temp.row.tolist()
        collist = temp.col.tolist()
        labellist = temp.label.tolist()

        # superpixels segmentation
        seg_path = imgpath + image_sets[i] + '_seg.npy'
        if path.exists(seg_path):
            seg = np.load(seg_path)
        else:
            seg = segmentation.quickshift(rgb_img, kernel_size=KERNEL_SIZE, max_dist=MAX_DIST, ratio=RATIO)
            np.save(seg_path, seg)

        seg = seg + 1
        props = regionprops(seg)

        for j in range(len(rowlist)):
            row_j = int(rowlist[j])
            col_j = int(collist[j])
            label_j = int(labellist[j])

            roi0, roi1, roi2 = create_multiscale_roi(rgb_img, seg, props, row_j, col_j)

            roi0 = cv2.resize(roi0, (img_h0, img_w0))
            roi1 = cv2.resize(roi1, (img_h1, img_w1))
            roi2 = cv2.resize(roi2, (img_h2, img_w2))

            # data augmentation
            roi0_aug_list = data_augment(roi0, NUM_AUGMENT)
            roi1_aug_list = data_augment(roi1, NUM_AUGMENT)
            roi2_aug_list = data_augment(roi2, NUM_AUGMENT)
            for k in range(len(roi0_aug_list)):
                cv2.imwrite(output + '0/' + str(count).zfill(7) + '.png', roi0_aug_list[k])
                cv2.imwrite(output + '1/' + str(count).zfill(7) + '.png', roi1_aug_list[k])
                cv2.imwrite(output + '2/' + str(count).zfill(7) + '.png', roi2_aug_list[k])

                images.append(image_sets[i])
                filenames.append(str(count).zfill(7) + '.png')
                rows.append(row_j)
                cols.append(col_j)
                labels.append(label_j)
                if k is 0:
                    is_original.append(1)
                else:
                    is_original.append(0)

                count += 1

    # save all.csv
    df = pandas.DataFrame(data={"image": images, "filename": filenames, "row": rows, "col": cols, "label": labels,
                                "is_original": is_original})
    df.to_csv(output + 'all.csv', sep=',', index=False)

    print('Create dataset done.\n')


def train_test_split(csvpath, test_rate=0.2):
    """
    Choose train and test set, then save the url into two csv files.
    :param csvpath: path to all.csv
    :param test_rate: test rate
    """
    print('Spliting dataset...')
    train_csv = pandas.read_csv(csvpath, names=['image', 'filename', 'row', 'col', 'label', 'is_original'], header=0)
    train_csv = train_csv.loc[train_csv['is_original'] == 1]
    train_csv = train_csv.sample(frac=1)  # shuffle
    image = train_csv.image.tolist()
    filename = train_csv.filename.tolist()
    row = train_csv.row.tolist()
    col = train_csv.col.tolist()
    label = train_csv.label.tolist()

    train_image = []
    train_filename = []
    train_row = []
    train_col = []
    train_label = []

    test_image = []
    test_filename = []
    test_row = []
    test_col = []
    test_label = []
    num_sample = len(train_csv)
    test_num = int(test_rate * num_sample)
    count = 0
    for i in range(len(filename)):
        if count < test_num:
            test_image.append(image[i])
            test_filename.append(filename[i])
            test_row.append(row[i])
            test_col.append(col[i])
            test_label.append(label[i])
            count = count + 1
        else:
            for j in range(NUM_AUGMENT):
                train_image.append(image[i])
                index = int(filename[i].split('.')[0]) + j
                train_filename.append(str(index).zfill(7) + '.png')
                train_row.append(row[i])
                train_col.append(col[i])
                train_label.append(label[i])

    # save and print result
    df1 = pandas.DataFrame(data={"image": train_image, "filename": train_filename, "row": train_row, "col": train_col,
                                 "label": train_label})
    df1.to_csv(output + 'train_list.csv', sep=',', index=False)
    df2 = pandas.DataFrame(
        data={"image": test_image, "filename": test_filename, "row": test_row, "col": test_col, "label": test_label})
    df2.to_csv(output + 'test_list.csv', sep=',', index=False)

    all_labels = np.unique(label)
    print('Train samples (before augmented)')
    for label in all_labels:
        num = int(train_label.count(label) / NUM_AUGMENT)
        print('label ' + str(label) + ' : ' + str(num))
    num = int(len(train_label) / NUM_AUGMENT)
    print('Total: ' + str(num))
    print('Test samples')
    for label in all_labels:
        num = test_label.count(label)
        print('label ' + str(label) + ' : ' + str(num))
    print('Total: ' + str(len(test_label)))
    print('Split train and test set done.\n')


if __name__ == '__main__':
    # imgpath = './data/src/'
    # labelpath = './data/label/'

    imgpath = '/home/ubuntu/Desktop/xiangliu/data/patch2/'
    csvpath = '/home/ubuntu/Desktop/xiangliu/csv/'

    # creat_csv_from_gt(imgpath=imgpath, labelpath=labelpath, num_sample=8000)
    creat_dataset_from_csv(imgpath=imgpath, csvpath=csvpath)
    train_test_split(csvpath=output + 'all.csv', test_rate=0.25)
