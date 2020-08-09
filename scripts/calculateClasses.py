import cv2
import numpy as np

filenames = [
    'top_mosaic_09cm_area1.tif',
    'top_mosaic_09cm_area2.tif',
    'top_mosaic_09cm_area3.tif',
    'top_mosaic_09cm_area4.tif',
    'top_mosaic_09cm_area5.tif',
    'top_mosaic_09cm_area6.tif',
    'top_mosaic_09cm_area7.tif',
    'top_mosaic_09cm_area8.tif',
    'top_mosaic_09cm_area10.tif',
    'top_mosaic_09cm_area26.tif'
]

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

if __name__ == '__main__':
    labels=[0,0,0,0,0,0]

    number = len(filenames)
    for i in range(number):
        img=cv2.imread(filenames[i])
        print('processing '+filenames[i])
        rows,cols,_=img.shape
        for r in range(rows):
            for c in range(cols):
                pixel=img[r,c,:]
                label=bgr2label(pixel)
                labels[label]=labels[label]+1

    sum=np.sum(labels)
    result=np.asarray(labels)/sum
    print(result)

