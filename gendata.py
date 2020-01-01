import cv2
import random
import pandas
import os
from os import path
from skimage import segmentation
from skimage.measure import regionprops
from numpy import expand_dims
import numpy as np

from keras.preprocessing.image import ImageDataGenerator


class DatasetGenerator(object):
    """Dataset generator object"""

    num_augment = 10
    img_w0 = 24
    img_h0 = 24
    img_w1 = 48
    img_h1 = 48
    img_w2 = 72
    img_h2 = 72
    seed = 1
    datagen = ImageDataGenerator(brightness_range=[0.3, 1.2],
                                 rotation_range=40,
                                 horizontal_flip=True,
                                 vertical_flip=True)

    def __init__(self, image_path_list, gt_path_list=None, samples_path_list=None):
        self.image_path_list = image_path_list
        self.gt_path_list = gt_path_list
        self.samples_path_list = samples_path_list
        self.from_gt = False
        self.from_csv = False
        self.output_dir = None

    @staticmethod
    def create(image_path_list, gt_path_list=None, samples_path_list=None):
        """
        Factory function, return instance if inputs are valid. Inputs should match each other.
        :param image_path_list: list, source image paths
        :param gt_path_list: list, ground truth image paths
        :param samples_path_list: list, sample csv paths
            csv format example:
            label,row,col
            0,2355,590
            0,1489,177
            2,946,571
        :return: A DatasetGenerator instance.
        """

        if gt_path_list is None and samples_path_list is None:
            raise Exception('Input invalid! Ground truth images or sample files are needed.')

        if gt_path_list is not None:
            if len(image_path_list) != len(gt_path_list):
                raise Exception('Input invalid! Images and ground truth images not matched.')
            else:
                obj = DatasetGenerator(image_path_list, gt_path_list, samples_path_list)
                obj.from_gt = True
                return obj

        if samples_path_list is not None:
            if len(image_path_list) != len(samples_path_list):
                raise Exception('Input invalid! Images and sample files not matched.')
            else:
                obj = DatasetGenerator(image_path_list, gt_path_list, samples_path_list)
                obj.from_csv = True
                return obj

    def creat_dataset(self, output_dir, num_sample=None):
        """
        Create dataset from gt or from csv.
        :param output_dir: output directory
        :param num_sample: number of samples in each image (if create dataset from ground truth images)
        :return:
        """
        self.output_dir = output_dir
        if os.path.isdir(output_dir + '0/') is False:
            os.mkdir(output_dir + '0/')
        if os.path.isdir(output_dir + '1/') is False:
            os.mkdir(output_dir + '1/')
        if os.path.isdir(output_dir + '2/') is False:
            os.mkdir(output_dir + '2/')

        if self.from_gt is True:
            self.create_from_gt(num_sample=num_sample)

            sample_list = []
            for i in range(len(self.image_path_list)):
                img_path = self.image_path_list[i]
                fname = img_path.split('/')[-1]
                sample_list.append(self.output_dir + fname + '.csv')
            self.samples_path_list = sample_list

            self.create_from_csv()
        elif self.from_csv is True:
            self.create_from_csv()
        else:
            raise Exception('DatasetGenerator is invalid. You need create an valid object first.')

    def train_test_split(self, test_rate=0.25):
        """
        Choose train and test set, then save the url into two csv files.
        :param test_rate: test rate
        """
        print('Spliting dataset...')
        train_csv = pandas.read_csv(self.output_dir + 'all.csv',
                                    names=['image', 'filename', 'row', 'col', 'label', 'is_original'],
                                    header=0)
        train_csv = train_csv.loc[train_csv['is_original'] == 1]
        train_csv = train_csv.sample(frac=1)  # shuffle
        image = train_csv.image.tolist()
        filename = train_csv.filename.tolist()
        row = train_csv.row.tolist()
        col = train_csv.col.tolist()
        label = train_csv.label.tolist()

        train_image, train_filename, train_row, train_col, train_label = [], [], [], [], []
        test_image, test_filename, test_row, test_col, test_label = [], [], [], [], []
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
                for j in range(self.num_augment):
                    train_image.append(image[i])
                    index = int(filename[i].split('.')[0]) + j
                    train_filename.append(str(index).zfill(7) + '.png')
                    train_row.append(row[i])
                    train_col.append(col[i])
                    train_label.append(label[i])

        # save and print result
        df1 = pandas.DataFrame(
            data={"image": train_image, "filename": train_filename, "row": train_row, "col": train_col,
                  "label": train_label})
        df1.to_csv(self.output_dir + 'train_list.csv', sep=',', index=False)
        df2 = pandas.DataFrame(
            data={"image": test_image, "filename": test_filename, "row": test_row, "col": test_col,
                  "label": test_label})
        df2.to_csv(self.output_dir + 'test_list.csv', sep=',', index=False)

        all_labels = np.unique(label)
        print('Train samples (before augmented)')
        for label in all_labels:
            num = int(train_label.count(label) / self.num_augment)
            print('label ' + str(label) + ' : ' + str(num))
        num = int(len(train_label) / self.num_augment)
        print('Total: ' + str(num))
        print('Test samples')
        for label in all_labels:
            num = test_label.count(label)
            print('label ' + str(label) + ' : ' + str(num))
        print('Total: ' + str(len(test_label)))
        print('Split train and test set done.\n')

    @staticmethod
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

    @staticmethod
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
        it = DatasetGenerator.datagen.flow(img, batch_size=number - 1, seed=DatasetGenerator.seed)
        for i in range(number - 1):
            batch = it.next()
            img_aug = batch[0].astype('uint8')
            img_aug_list.append(img_aug)
        return img_aug_list

    @staticmethod
    def bgr2label(pixel, dataset_type):
        pixel = pixel.tolist()

        if dataset_type is 'vaihingen':
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
        elif dataset_type is 'xiangliu':
            if pixel == [0, 0, 0]:  # nothing
                return 0
            elif pixel == [209, 111, 100]:  # Floating plants
                return 1
            elif pixel == [229, 229, 229]:  # Roads
                return 2
            elif pixel == [168, 211, 58]:  # Crops
                return 3
            elif pixel == [51, 160, 44]:  # Trees
                return 4
            elif pixel == [170, 255, 91]:  # Shrubs
                return 5
            elif pixel == [253, 191, 111]:  # Bare soil
                return 6
            elif pixel == [128, 79, 193]:  # Buildings
                return 7
            elif pixel == [67, 177, 213]:  # Water
                return 8

    @staticmethod
    def create_patch_roi(img, r_center, c_center, patch_size):
        """
        Create patch-based roi, i.e. pixel-wise.
        Single scale.
        :param img: original image
        :param r_center: row index
        :param c_center: col index
        :return: roi image
        """
        rows, cols, _ = img.shape
        margin = int(patch_size / 2)
        sub_img = img[max(r_center - margin, 0):min(r_center + margin, rows - 1),
                  max(c_center - margin, 0): min(c_center + margin, cols - 1), :]
        return sub_img

    @staticmethod
    def create_patch_roi_v2(img, r_upleft, c_upleft, patch_size):
        """
        Create patch-based roi, i.e. pixel-wise.
        :param img: original image
        :param r_upleft: patch's up-left row index
        :param c_upleft: patch's up-left col index
        :param patch_size: patch size
        :return: roi image
        """

        rows, cols, _ = img.shape
        sub_img = img[r_upleft:min(r_upleft + patch_size, rows - 1),
                  c_upleft:min(c_upleft + patch_size, cols - 1), :]

        return sub_img

    @staticmethod
    def create_singlescale_roi(img, seg, props, row, col, scale):
        """
        Create objected-oriented roi, single scale.

        :param img: original image
        :param seg: segmentation image
        :param props: regionprops object
        :param row: sample row index
        :param col: sample col index
        :return: roi image
        :param scale: object input scale, this should match with the scale of the network
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

        if scale is 0:
            return sub_img_0
        elif scale is 1:
            return sub_img_1
        else:
            return sub_img_2

    @staticmethod
    def segmentation(image_path, kernel_size=3, max_dist=6, ratio=0.5):
        """
        Quickshift superpixels segmentation.
        If previous segmentation result already exist in current folder, load it directly.
        :return: segmentation object
        """
        rgb_img = cv2.imread(image_path)
        seg_path = image_path + '_seg.npy'
        if path.exists(seg_path):
            seg = np.load(seg_path)
        else:
            seg = segmentation.quickshift(rgb_img, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
            np.save(seg_path, seg)

        return seg

    def create_from_gt(self, num_sample):
        """
        Create dataset from random samples, use when you already have a ground truth.
        Note that some sample points may fall into the same object, this will be tackled in  creat_dataset_from_csv function.
        Save all.csv and dataset in train folder.
        :param num_sample: number of samples in each image
        :return:
        """

        print('Creating csv...')

        for i in range(len(self.image_path_list)):
            # load image and label (both in 3 channels)
            img_path = self.image_path_list[i]
            label_path = self.gt_path_list[i]
            rgb_img = cv2.imread(img_path)
            label_img = cv2.imread(label_path)
            print('Creating csv from ' + img_path)

            row_list, col_list, label_list = [], [], []
            # random sample
            sample_margin = 20
            rows, cols, _ = rgb_img.shape
            for j in range(num_sample):
                row_j = random.randint(sample_margin, rows - sample_margin)
                col_j = random.randint(sample_margin, cols - sample_margin)
                label_j = self.bgr2label(label_img[row_j, col_j, :], dataset_type='vaihingen')
                row_list.append(row_j)
                col_list.append(col_j)
                label_list.append(label_j)

            df = pandas.DataFrame(data={'label': label_list, 'row': row_list, 'col': col_list})
            fname = img_path.split('/')[-1]
            df.to_csv(self.output_dir + fname + '.csv', sep=',', index=False)

    def create_from_csv(self):
        """
        Create dataset from manual sample point file (.csv format)
        Use when you don't have a ground truth, this will save all.csv and dataset in train folder.
        """

        print('Creating dataset from csv...')

        count = 0
        images, filenames, rows, cols, labels, is_original = [], [], [], [], [], []
        for i in range(len(self.image_path_list)):
            rgb_path = self.image_path_list[i]
            rgb_img = cv2.imread(rgb_path)
            print('Creating dataset from ' + rgb_path)

            csv = pandas.read_csv(self.samples_path_list[i], names=['label', 'row', 'col'], header=0)
            csv = csv.dropna()
            rowlist = csv.row.tolist()
            collist = csv.col.tolist()
            labellist = csv.label.tolist()

            seg = self.segmentation(image_path=rgb_path)
            seg = seg + 1
            props = regionprops(seg)

            for j in range(len(rowlist)):
                row_j = int(rowlist[j])
                col_j = int(collist[j])
                label_j = int(labellist[j])

                roi0, roi1, roi2 = self.create_multiscale_roi(rgb_img, seg, props, row_j, col_j)

                roi0 = cv2.resize(roi0, (self.img_h0, self.img_w0))
                roi1 = cv2.resize(roi1, (self.img_h1, self.img_w1))
                roi2 = cv2.resize(roi2, (self.img_h2, self.img_w2))

                # data augmentation
                roi0_aug_list = self.data_augment(roi0, self.num_augment)
                roi1_aug_list = self.data_augment(roi1, self.num_augment)
                roi2_aug_list = self.data_augment(roi2, self.num_augment)
                for k in range(len(roi0_aug_list)):
                    cv2.imwrite(self.output_dir + '0/' + str(count).zfill(7) + '.png', roi0_aug_list[k])
                    cv2.imwrite(self.output_dir + '1/' + str(count).zfill(7) + '.png', roi1_aug_list[k])
                    cv2.imwrite(self.output_dir + '2/' + str(count).zfill(7) + '.png', roi2_aug_list[k])

                    fname = rgb_path.split('/')[-1]
                    images.append(fname)
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
        df.to_csv(self.output_dir + 'all.csv', sep=',', index=False)

        print('Create dataset done.\n')


if __name__ == '__main__':
    # image_path_list = [
    #     '/home/ubuntu/Desktop/xiangliu/data/patch2/2.tif',
    #     '/home/ubuntu/Desktop/xiangliu/data/patch2/3.tif',
    #     '/home/ubuntu/Desktop/xiangliu/data/patch2/8.tif',
    #     '/home/ubuntu/Desktop/xiangliu/data/patch2/9.tif',
    #     '/home/ubuntu/Desktop/xiangliu/data/patch2/14.tif',
    #     '/home/ubuntu/Desktop/xiangliu/data/patch2/16.tif',
    #     '/home/ubuntu/Desktop/xiangliu/data/patch2/20.tif',
    #     '/home/ubuntu/Desktop/xiangliu/data/patch2/22.tif',
    #     '/home/ubuntu/Desktop/xiangliu/data/patch2/26.tif',
    #     '/home/ubuntu/Desktop/xiangliu/data/patch2/28.tif',
    #     '/home/ubuntu/Desktop/xiangliu/data/patch2/33.tif'
    # ]
    # samples_path_list = [
    #     '/home/ubuntu/Desktop/xiangliu/csv/2.tif.csv',
    #     '/home/ubuntu/Desktop/xiangliu/csv/3.tif.csv',
    #     '/home/ubuntu/Desktop/xiangliu/csv/8.tif.csv',
    #     '/home/ubuntu/Desktop/xiangliu/csv/9.tif.csv',
    #     '/home/ubuntu/Desktop/xiangliu/csv/14.tif.csv',
    #     '/home/ubuntu/Desktop/xiangliu/csv/16.tif.csv',
    #     '/home/ubuntu/Desktop/xiangliu/csv/20.tif.csv',
    #     '/home/ubuntu/Desktop/xiangliu/csv/22.tif.csv',
    #     '/home/ubuntu/Desktop/xiangliu/csv/26.tif.csv',
    #     '/home/ubuntu/Desktop/xiangliu/csv/28.tif.csv',
    #     '/home/ubuntu/Desktop/xiangliu/csv/33.tif.csv'
    # ]
    # dataset_gen = DatasetGenerator.create(image_path_list=image_path_list, samples_path_list=samples_path_list)
    # dataset_gen.creat_dataset(output_dir='./data/train/')
    # dataset_gen.train_test_split(test_rate=0.25)

    img_path_lst = [
        './data/src/top_mosaic_09cm_area1.tif',
        './data/src/top_mosaic_09cm_area2.tif'
    ]

    gt_path_lst = [
        './data/label/top_mosaic_09cm_area1.tif',
        './data/label/top_mosaic_09cm_area2.tif'
    ]

    dataset_gen2 = DatasetGenerator.create(image_path_list=img_path_lst, gt_path_list=gt_path_lst)
    dataset_gen2.creat_dataset(output_dir='./data/train/', num_sample=8000)
    dataset_gen2.train_test_split(test_rate=0.25)
