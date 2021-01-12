import cv2
import numpy as np
from os import path

from skimage import segmentation
from skimage.color import rgb2gray
from skimage.filters import sobel

def quickshift(image_path, kernel_size=3, max_dist=6, ratio=0.5):
    """
    Quickshift superpixels segmentation.
    If previous segmentation result already exist in current folder, load it directly.
    :return: segmentation object
    """
    rgb_img = cv2.imread(image_path)
    seg_path = image_path + '_seg_quickshift.npy'
    if path.exists(seg_path):
        seg = np.load(seg_path)
    else:
        seg = segmentation.quickshift(rgb_img, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
        np.save(seg_path, seg)

    seg = seg + 1  # quickshift seg start from zero
    return seg


def slic(image_path):
    """
    SLIC superpixels segmentation.
    If previous segmentation result already exist in current folder, load it directly.
    :return: segmentation object
    """
    rgb_img = cv2.imread(image_path)
    seg_path = image_path + '_seg_slic.npy'
    if path.exists(seg_path):
        seg = np.load(seg_path)
    else:
        seg = segmentation.slic(rgb_img, n_segments=18110, compactness=10, sigma=1, start_label=1)
        np.save(seg_path, seg)

    return seg


def watershed(image_path):
    """
    Watershed superpixels segmentation.
    If previous segmentation result already exist in current folder, load it directly.
    :return: segmentation object
    """
    rgb_img = cv2.imread(image_path)
    seg_path = image_path + '_seg_watershed.npy'
    if path.exists(seg_path):
        seg = np.load(seg_path)
    else:
        gradient = sobel(rgb2gray(rgb_img))
        seg = segmentation.watershed(gradient, markers=16100, compactness=0.001)
        np.save(seg_path, seg)

    return seg