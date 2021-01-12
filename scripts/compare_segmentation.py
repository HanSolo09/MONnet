import cv2
import time
import numpy as np

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_ubyte


def test_felzenszwalb(image):
    start_time = time.time()
    seg = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    end_time = time.time()
    cost_time = end_time - start_time
    num_seg = len(np.unique(seg))

    print("Felzenszwalb number of segments: %d, time cost: %fs" % (num_seg, cost_time))
    cv2.imwrite("Felzenszwalbs.png", img_as_ubyte(mark_boundaries(image, seg, (0, 1, 1))))

    return seg


def test_slic(image):
    start_time = time.time()
    seg = slic(image, n_segments=18110, compactness=10, sigma=1, start_label=1)  # n_segments should be set as a small value (such as 200) for small-size image
    end_time = time.time()
    cost_time = end_time - start_time
    num_seg = len(np.unique(seg))

    print("SLIC number of segments: %d, time cost: %fs" % (num_seg, cost_time))

    return seg


def test_quickshift(image):
    start_time = time.time()
    seg = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
    end_time = time.time()
    cost_time = end_time - start_time
    num_seg = len(np.unique(seg))

    print("Quickshift number of segments: %d, time cost: %fs" % (num_seg, cost_time))
    cv2.imwrite("Quickshift.png", img_as_ubyte(mark_boundaries(image, seg, (0, 1, 1))))

    return seg


def test_watershed(image):
    start_time = time.time()
    gradient = sobel(rgb2gray(image))
    seg = watershed(gradient, markers=16100, compactness=0.001)  # markers should be set as a small value (such as 200) for small-size image
    end_time = time.time()
    cost_time = end_time - start_time
    num_seg = len(np.unique(seg))

    print("Watershed number of segments: %d, time cost: %fs" % (num_seg, cost_time))
    cv2.imwrite("Watershed.png", img_as_ubyte(mark_boundaries(image, seg, (0, 1, 1))))

    return seg


if __name__ == '__main__':
    image_path = '/home/irsgis/data/MONet_data/image/top_mosaic_09cm_area4.tif'
    img = cv2.imread(image_path)

    # slice a ROI
    start = 1040
    end = 1240

    seg_slic = test_slic(img)
    cv2.imwrite("SLIC.png", img_as_ubyte(mark_boundaries(img, seg_slic, (0, 1, 1))))
    cv2.imwrite("SLIC_roi.png", img_as_ubyte(mark_boundaries(img[start:end, start:end, :], seg_slic[start:end, start:end], (0, 1, 1))))

    seg_quickshift = test_quickshift(img)
    cv2.imwrite("Quickshift.png", img_as_ubyte(mark_boundaries(img, seg_quickshift, (0, 1, 1))))
    cv2.imwrite("Quickshift_roi.png", img_as_ubyte(mark_boundaries(img[start:end, start:end, :], seg_quickshift[start:end, start:end], (0, 1, 1))))

    seg_watershed = test_watershed(img)
    cv2.imwrite("Watershed.png", img_as_ubyte(mark_boundaries(img, seg_watershed, (0, 1, 1))))
    cv2.imwrite("Watershed_roi.png", img_as_ubyte(mark_boundaries(img[start:end, start:end, :], seg_watershed[start:end, start:end], (0, 1, 1))))
