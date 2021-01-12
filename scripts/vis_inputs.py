import sys
from skimage.segmentation import mark_boundaries
from gen_data import *

sys.path.append('../utils/')

image_sets = [
    'top_mosaic_09cm_area1.tif'
    # 'top_mosaic_09cm_area2.tif',
    # 'top_mosaic_09cm_area3.tif',
    # 'top_mosaic_09cm_area4.tif',
    # 'top_mosaic_09cm_area5.tif',
    # 'top_mosaic_09cm_area6.tif',
    # 'top_mosaic_09cm_area7.tif',
    # 'top_mosaic_09cm_area8.tif',
    # 'top_mosaic_09cm_area10.tif',
    # 'top_mosaic_09cm_area26.tif'
]
imgpath = '/home/ubuntu/data/mcnn_data/src/'

img_w0 = 24
img_h0 = 24
img_w1 = 48
img_h1 = 48
img_w2 = 72
img_h2 = 72

filepath = imgpath + image_sets[0]
rgb_img = cv2.imread(filepath)
rows, cols, _ = rgb_img.shape
print("process " + filepath)
print("rows " + str(rows) + " , cols " + str(cols))

seg = DatasetGenerator.segmentation_quickshift(image_path=filepath)
seg = seg + 1

# slice a ROI
start = 1040
end = 1240
mid = (end - start) // 2

rgb_img = rgb_img[start:end, start:end, :]
seg = seg[start:end, start:end]

# draw boundaries
bound = mark_boundaries(rgb_img, seg, (0, 1, 1))
bound = bound * 255
bound = bound.astype(np.uint8)
cv2.imwrite('overview0.png', bound)

# choose one segment
center_label = seg[mid, mid]
seg[seg[:, :] != center_label] = 0
seg[seg[:, :] == center_label] = 1

# draw boundaries
bound = mark_boundaries(rgb_img, seg, (0, 1, 1))
bound = bound * 255
bound = bound.astype(np.uint8)
cv2.imwrite('overview.png', bound)

props = regionprops(seg)
row_j = props[0].coords[0][0]
col_j = props[0].coords[0][1]

# create multiscale input
roi0, roi1, roi2 = DatasetGenerator.create_multiscale_roi(bound, seg, props, row_j, col_j)
cv2.imwrite('s1.png', roi0)
cv2.imwrite('s2.png', roi1)
cv2.imwrite('s3.png', roi2)
roi0 = cv2.resize(roi0, (img_h0, img_w0))
roi1 = cv2.resize(roi1, (img_h1, img_w1))
roi2 = cv2.resize(roi2, (img_h2, img_w2))
cv2.imwrite('s1_resize.png', roi0)
cv2.imwrite('s2_resize.png', roi1)
cv2.imwrite('s3_resize.png', roi2)
