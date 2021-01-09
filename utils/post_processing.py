from skimage import measure
from collections import Counter
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


def draw(img, dataset_type):
    """
    dimging color prediction result according to its dataset type.
    :param img: original prediction image (single channel)
    :param dataset_type: dataset type. Should be either 'vaihingen' or 'xiangliu'
    :return: colored prediction image (3 channels)
    """
    row, col = img.shape
    res = np.zeros((row, col, 3), np.uint8)
    if dataset_type is 'vaihingen':
        for i in range(row):
            for j in range(col):
                pixel = img[i, j].tolist()
                if pixel == 0:
                    res[i, j] = [255, 255, 255]  # Road
                elif pixel == 1:
                    res[i, j] = [0, 0, 255]  # Water
                elif pixel == 2:
                    res[i, j] = [255, 0, 0]  # Buildings
                elif pixel == 3:
                    res[i, j] = [0, 255, 255]  # Cars
                elif pixel == 4:
                    res[i, j] = [0, 255, 0]  # Trees
                elif pixel == 5:
                    res[i, j] = [255, 255, 0]  # Grass
    elif dataset_type is 'xiangliu':
        for i in range(row):
            for j in range(col):
                pixel = img[i, j].tolist()
                if pixel == 0:
                    res[i, j] = [0, 0, 0]  # nothing
                elif pixel == 1:
                    res[i, j] = [100, 111, 209]  # Floating plants
                elif pixel == 2:
                    res[i, j] = [229, 229, 229]  # Roads
                elif pixel == 3:
                    res[i, j] = [58, 211, 168]  # Crops
                elif pixel == 4:
                    res[i, j] = [44, 160, 51]  # Trees
                elif pixel == 5:
                    res[i, j] = [91, 255, 170]  # Shrubs
                elif pixel == 6:
                    res[i, j] = [111, 191, 253]  # Bare soil
                elif pixel == 7:
                    res[i, j] = [193, 79, 128]  # Buildings
                elif pixel == 8:
                    res[i, j] = [213, 177, 67]  # Water

    return res


def crf(im, mask, gt_prob=.9, sxy_gaussian=3, sxy_bilateral=60, srgb_bilateral=10):
    """
    Fully connected CRF post processing function
    :param im: original image (single channel or 3 channels)
    :param mask: your original prediction result (single channel)
    :return: crf post processing image (single channel or 3 channels)
    """
    colors, labels = np.unique(mask, return_inverse=True)
    image_size = mask.shape[:2]
    n_labels = len(set(labels.flat))
    d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)  # width, height, nlabels
    U = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=3)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=im.astype('uint8'), compat=10)
    Q = d.inference(5)  # 5 - num of iterations
    MAP = np.argmax(Q, axis=0).reshape(image_size)
    unique_map = np.unique(MAP)
    result = np.zeros((image_size[0], image_size[1]), np.uint8)
    for u in unique_map:  # get original labels back
        np.putmask(result, MAP == u, colors[u])
    return result


def vote(img, seg):
    """
    Object-based voting.
    :param img: original prediction image (single channel)
    :param seg: object segmentation image
    :return: processed result (single channel)
    """
    res = seg.copy()
    props = measure.regionprops(seg)
    for region in props:
        # print(region.label)
        coord = region.coords
        labels = img[coord[:, 0], coord[:, 1]]
        counts = Counter(labels)
        the_label = counts.most_common(1)[0][0]
        res[coord[:, 0], coord[:, 1]] = the_label

    return res
