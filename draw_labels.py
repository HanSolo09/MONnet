import cv2
import numpy as np


def draw_labels(filename):
    img = cv2.imread(filename, -1)
    row, col = img.shape
    res = np.zeros((row, col, 3), np.uint8)
    for i in range(row):
        for j in range(col):
            pixel = img[i, j].tolist()
            if pixel == 0:
                res[i, j] = [255, 255, 255]
            elif pixel == 1:
                res[i, j] = [0, 0, 255]
            elif pixel == 2:
                res[i, j] = [255, 0, 0]
            elif pixel == 3:
                res[i, j] = [0, 255, 255]
            elif pixel == 4:
                res[i, j] = [0, 255, 0]
            elif pixel == 5:
                res[i, j] = [255, 255, 0]

    cv2.imwrite('./data/predict/visualize.png', res)


if __name__ == '__main__':
    draw_labels('./data/predict/0.png')
