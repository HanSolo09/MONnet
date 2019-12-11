import cv2
import random
import pandas
from skimage import segmentation
from skimage.measure import regionprops
from numpy import expand_dims

from keras.preprocessing.image import ImageDataGenerator

# make sure these folders exist
output = './data/train/'
image_sets = [
    'top_mosaic_09cm_area1.tif'
    'top_mosaic_09cm_area2.tif',
    'top_mosaic_09cm_area3.tif',
    'top_mosaic_09cm_area4.tif',
    'top_mosaic_09cm_area5.tif',
    'top_mosaic_09cm_area6.tif',
    'top_mosaic_09cm_area7.tif',
    'top_mosaic_09cm_area8.tif',
    'top_mosaic_09cm_area10.tif',
    'top_mosaic_09cm_area26.tif'

    # '2',
    # '3',
    # '8',
    # '9',
    # '14',
    # '16',
    # '20',
    # '22',
    # '26',
    # '28'
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
datagen = ImageDataGenerator(brightness_range=[0.2, 1.0],
                             rotation_range=90,
                             horizontal_flip=True,
                             vertical_flip=True)


def data_augment(img, number):
    img_aug_list = []
    img_aug_list.append(img)  # include origin image
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
    margin = img_w0 / 2
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


def creat_dataset_from_gt(imgpath, labelpath, num_sample=5000):
    """
    Create dataset from random samples, use when you already have a ground truth.
    Save train.csv and dataset in train folder.

    :param imgpath: source image path
    :param labelpath: ground truth label image path
    :param num_sample: number of samples in each image
    """
    print('creating dataset...')

    count = 0
    images = []
    filenames = []
    rows = []
    cols = []
    labels = []
    is_original = []
    for i in range(len(image_sets)):
        # load image and label (both in 3 channels)
        print('creating dataset from image ' + str(i))
        rgb_img = cv2.imread(imgpath + image_sets[i])
        row, col, _ = rgb_img.shape

        src_img = rgb_img
        label_img = cv2.imread(labelpath + image_sets[i])

        # superpixels segmentation
        seg = segmentation.quickshift(rgb_img, kernel_size=KERNEL_SIZE, max_dist=MAX_DIST, ratio=RATIO)
        seg = seg + 1
        props = regionprops(seg)

        # random sample
        sample_margin = 20
        for j in range(num_sample):
            row_j = random.randint(sample_margin, row - sample_margin)
            col_j = random.randint(sample_margin, col - sample_margin)
            label_j = bgr2label(label_img[row_j, col_j, :])

            roi0, roi1, roi2 = create_multiscale_roi(src_img, seg, props, row_j, col_j)

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

    # save train.csv
    df = pandas.DataFrame(data={"image": images, "filename": filenames, "row": rows, "col": cols, "label": labels,
                                "is_original": is_original})
    df.to_csv(output + 'train.csv', sep=',', index=False)

    print('Create dataset done')


def creat_dataset_from_samples(imgpath, csvpath):
    """
    Create dataset from manual sample point (.csv format)
    Use when you don't have a ground truth, this will save train.csv and dataset in train folder.

    :param imgpath: source image path
    :param csvpath: sample csv path
    """
    print('creating dataset...')

    count = 0
    images = []
    filenames = []
    rows = []
    cols = []
    labels = []
    is_original = []
    for i in range(len(image_sets)):
        print('creating dataset from image ' + str(i))
        rgb_img = cv2.imread(imgpath + image_sets[i] + '.tif')

        row, col, _ = rgb_img.shape

        # sample from file
        temp = pandas.read_csv(csvpath + image_sets[i] + '.csv', names=['label', 'row', 'col'], header=0)
        temp = temp.dropna()
        rowlist = temp.row.tolist()
        collist = temp.col.tolist()
        labellist = temp.label.tolist()

        # superpixels segmentation
        seg = segmentation.quickshift(rgb_img, kernel_size=KERNEL_SIZE, max_dist=MAX_DIST, ratio=RATIO)
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

                images.append(image_sets[i] + '.tif')
                filenames.append(str(count).zfill(7) + '.png')
                rows.append(row_j)
                cols.append(col_j)
                labels.append(label_j)
                if k is 0:
                    is_original.append(1)
                else:
                    is_original.append(0)

                count += 1

    # save train.csv
    df = pandas.DataFrame(data={"image": images, "filename": filenames, "row": rows, "col": cols, "label": labels,
                                "is_original": is_original})
    df.to_csv(output + 'train.csv', sep=',', index=False)

    print('Create dataset done')


def train_test_split(csvpath, test_rate=0.1):
    """
    Choose train and test set, then save the url into two csv files.
    :param csvpath: path to train.csv
    :param test_rate: test rate
    """
    train_csv = pandas.read_csv(csvpath, names=['image', 'filename', 'row', 'col', 'label', 'is_original'], header=0)
    train_csv = train_csv.loc[train_csv['is_original'] == 1]
    train_csv = train_csv.sample(frac=1)  # shuffle
    image = train_csv.image.tolist()
    filename = train_csv.filename.tolist()
    row = train_csv.row.tolist()
    col = train_csv.col.tolist()
    label = train_csv.label.tolist()

    train_filename = []
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
                index = int(filename[i].split('.')[0]) + j
                train_filename.append(str(index).zfill(7) + '.png')
                train_label.append(label[i])

    df1 = pandas.DataFrame(data={"filename": train_filename, "label": train_label})
    df1.to_csv(output + 'train_list.csv', sep=',', index=False)
    df2 = pandas.DataFrame(
        data={"image": test_image, "filename": test_filename, "row": test_row, "col": test_col, "label": test_label})
    df2.to_csv(output + 'test_list.csv', sep=',', index=False)

    print('Create train and test set done')


if __name__ == '__main__':
    imgpath = './data/src/'
    labelpath = './data/label/'
    creat_dataset_from_gt(imgpath=imgpath, labelpath=labelpath, num_sample=10000)

    # imgpath = '/home/ubuntu/Desktop/xiangliu/data/patch2/'
    # csvpath = '/home/ubuntu/Desktop/xiangliu/csv/'
    # creat_dataset_from_samples(imgpath=imgpath, csvpath=csvpath)

    train_test_split(csvpath=output + 'train.csv', test_rate=0.2)
