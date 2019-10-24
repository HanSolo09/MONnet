import cv2
import numpy as np

filepath = './data/predict/'
img_sets = [
    'top_mosaic_09cm_area1.tif',
    'top_mosaic_09cm_area2.tif',
    'top_mosaic_09cm_area3.tif',
    'top_mosaic_09cm_area4.tif',
    'top_mosaic_09cm_area5.tif',
    'top_mosaic_09cm_area6.tif',
    'top_mosaic_09cm_area7.tif',
    'top_mosaic_09cm_area8.tif',
    'top_mosaic_09cm_area10.tif',
    'top_mosaic_09cm_area26.tif',
    'top_mosaic_09cm_area27.tif'
]


def draw_labels(filename):
    img = cv2.imread(filepath + filename, -1)
    row, col = img.shape
    res = np.zeros((row, col, 3), np.uint8)
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

    cv2.imwrite(filepath + filename.split('.')[0] + '_vis.png', res)


if __name__ == '__main__':
    for img in img_sets:
        draw_labels(img.split('.')[0] + '_pred.png')
