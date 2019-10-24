import os
import cv2
import random
import pandas
from skimage import segmentation
from skimage.measure import regionprops
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

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

    '2',
    '3',
    '8',
    '9',
    '14',
    '20'
]

sample_margin = 20
num_segmentaion = 20000

KERNEL_SIZE = 3
MAX_DIST = 6
RATIO = 0.5

NUM_SAMPLE = 5000
img_w0 = 24
img_h0 = 24
img_w1 = 48
img_h1 = 48
img_w2 = 72
img_h2 = 72

VALIDATION_RATE = 0.1

seg_id_list = []

datagen = ImageDataGenerator(brightness_range=[0.2, 1.0],
                             rotation_range=90,
                             horizontal_flip=True,
                             vertical_flip=True)

seed=1

def data_augment(img):
    img = expand_dims(img, 0)
    it = datagen.flow(img, batch_size=9, seed=seed)
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


# create objected-oriented roi
def create_multiscale_roi(img, seg, props, row, col):
    seg_id = seg[row, col] - 1
    # if seg_id in seg_id_list:
    #     return np.array([]), np.array([]), np.array([])
    # else:
    #     seg_id_list.append(seg_id)
    #
    #     bbox = props[seg_id].bbox
    #     min_row = bbox[0]
    #     min_col = bbox[1]
    #     max_row = bbox[2]
    #     max_col = bbox[3]
    #
    #     row, col, _ = img.shape
    #
    #     # scale 0
    #     sub_img_0 = img[min_row:max_row, min_col: max_col, :]
    #
    #     # scale 1
    #     margin1 = 15
    #     sub_img_1 = img[max(min_row - margin1, 0):min(max_row + margin1, row - 1),
    #                 max(min_col - margin1, 0): min(max_col + margin1, col - 1), :]
    #
    #     # scale 2
    #     margin2 = 30
    #     sub_img_2 = img[max(min_row - margin2, 0):min(max_row + margin2, row - 1),
    #                 max(min_col - margin2, 0): min(max_col + margin2, col - 1), :]
    #
    #     return sub_img_0, sub_img_1, sub_img_2

    seg_id_list.append(seg_id)

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


# create dataset from random samples
def creat_dataset(num_sample=NUM_SAMPLE):
    print('creating dataset...')
    count = 0
    images = []
    filenames = []
    rows = []
    cols = []
    labels = []
    for i in range(len(image_sets)):
        # load image and label (both in 3 channels)
        print('creating dataset from image ' + str(i))
        rgb_img = cv2.imread('./data/src/' + image_sets[i])
        row, col, _ = rgb_img.shape

        src_img = rgb_img
        label_img = cv2.imread('./data/label/' + image_sets[i])

        # superpixels segmentation
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
            if roi0.size == 0:
                continue

            # save origin patch
            roi0 = cv2.resize(roi0, (img_h0, img_w0))
            roi1 = cv2.resize(roi1, (img_h1, img_w1))
            roi2 = cv2.resize(roi2, (img_h2, img_w2))
            cv2.imwrite(output + '0/' + str(count).zfill(7) + '.png', roi0.astype('uint8'))
            cv2.imwrite(output + '1/' + str(count).zfill(7) + '.png', roi1.astype('uint8'))
            cv2.imwrite(output + '2/' + str(count).zfill(7) + '.png', roi2.astype('uint8'))
            images.append(image_sets[i])
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
                roi0_aug_list[k] = cv2.resize(roi0_aug_list[k], (img_h0, img_w0))
                roi1_aug_list[k] = cv2.resize(roi1_aug_list[k], (img_h1, img_w1))
                roi2_aug_list[k] = cv2.resize(roi2_aug_list[k], (img_h2, img_w2))
                cv2.imwrite(output + '0/' + str(count).zfill(7) + '.png', roi0_aug_list[k])
                cv2.imwrite(output + '1/' + str(count).zfill(7) + '.png', roi1_aug_list[k])
                cv2.imwrite(output + '2/' + str(count).zfill(7) + '.png', roi2_aug_list[k])
                images.append(image_sets[i])
                filenames.append(str(count).zfill(7) + '.png')
                rows.append(row_j)
                cols.append(col_j)
                labels.append(label_j)
                count += 1

    df = pandas.DataFrame(data={"Image": images, "filename": filenames, "row": rows, "col": cols, "label": labels})
    df.to_csv(output + 'train.csv', sep=',', index=False)

    # choose train and validation set
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(output + '0'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(VALIDATION_RATE * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])

    df2 = pandas.DataFrame(data={"train_list": train_set})
    df2.to_csv(output + 'train_list.csv', sep=',', index=False)
    df3 = pandas.DataFrame(data={"validation_list": val_set})
    df3.to_csv(output + 'validation_list.csv', sep=',', index=False)

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


# create dataset from manual sample point
def creat_dataset_v2(imgpath, csvpath):
    print('creating dataset...')
    count = 0
    images = []
    filenames = []
    rows = []
    cols = []
    labels = []
    for i in range(len(image_sets)):
        print('creating dataset from image ' + str(i))
        rgb_img = cv2.imread(imgpath + image_sets[i] + '.tif')

        row, col, _ = rgb_img.shape

        # sample from file
        temp = pandas.read_csv(csvpath + image_sets[i] + '.csv', names=['label', 'row', 'col'])
        temp = temp.dropna()
        rowlist = temp.row.tolist()
        collist = temp.col.tolist()
        labellist = temp.label.tolist()
        rowlist.pop(0)
        collist.pop(0)
        labellist.pop(0)

        # superpixels segmentation
        seg = segmentation.quickshift(rgb_img, kernel_size=KERNEL_SIZE, max_dist=MAX_DIST, ratio=RATIO)
        # seg = segmentation.slic(src_img, num_segmentaion)
        seg = seg + 1
        props = regionprops(seg)

        for j in range(len(rowlist)):
            row_j = int(rowlist[j])
            col_j = int(collist[j])
            label_j = int(labellist[j])

            roi0, roi1, roi2 = create_multiscale_roi(rgb_img, seg, props, row_j, col_j)
            # if roi0.size == 0:
            #     continue

            # save origin patch
            roi0 = cv2.resize(roi0, (img_h0, img_w0))
            roi1 = cv2.resize(roi1, (img_h1, img_w1))
            roi2 = cv2.resize(roi2, (img_h2, img_w2))
            cv2.imwrite(output + '0/' + str(count).zfill(7) + '.png', roi0.astype('uint8'))
            cv2.imwrite(output + '1/' + str(count).zfill(7) + '.png', roi1.astype('uint8'))
            cv2.imwrite(output + '2/' + str(count).zfill(7) + '.png', roi2.astype('uint8'))
            images.append(image_sets[i] + '.tif')
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
                roi0_aug_list[k] = cv2.resize(roi0_aug_list[k], (img_h0, img_w0))
                roi1_aug_list[k] = cv2.resize(roi1_aug_list[k], (img_h1, img_w1))
                roi2_aug_list[k] = cv2.resize(roi2_aug_list[k], (img_h2, img_w2))
                cv2.imwrite(output + '0/' + str(count).zfill(7) + '.png', roi0_aug_list[k])
                cv2.imwrite(output + '1/' + str(count).zfill(7) + '.png', roi1_aug_list[k])
                cv2.imwrite(output + '2/' + str(count).zfill(7) + '.png', roi2_aug_list[k])
                images.append(image_sets[i] + '.tif')
                filenames.append(str(count).zfill(7) + '.png')
                rows.append(row_j)
                cols.append(col_j)
                labels.append(label_j)
                count += 1

    df = pandas.DataFrame(data={"Image": images, "filename": filenames, "row": rows, "col": cols, "label": labels})
    df.to_csv(output + 'train.csv', sep=',', index=False)

    # choose train and validation set
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(output + '0'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(VALIDATION_RATE * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])

    df2 = pandas.DataFrame(data={"train_list": train_set})
    df2.to_csv(output + 'train_list.csv', sep=',', index=False)
    df3 = pandas.DataFrame(data={"validation_list": val_set})
    df3.to_csv(output + 'validation_list.csv', sep=',', index=False)

    with open(output + 'param.txt', 'w') as f:
        f.write('KERNEL_SIZE: ' + str(KERNEL_SIZE) + '\n')
        f.write('MAX_DIST: ' + str(MAX_DIST) + '\n')
        f.write('RATIO: ' + str(RATIO) + '\n')
        f.write('NUM_SEGMENTATION: ' + str(num_segmentaion) + '\n')
        f.write('IMAGE_SETS: \n')
        for image in image_sets:
            f.write(image + '\n')
        f.close()

    print('All saved')


if __name__ == '__main__':
    # creat_dataset()

    imgpath = '/home/ubuntu/Desktop/xiangliu/data/patch2/'
    csvpath = '/home/ubuntu/Desktop/xiangliu/csv/'
    creat_dataset_v2(imgpath, csvpath)
