from gen_data import *

csv_dir='/home/ubuntu/data/mcnn_data/vaihingen_final/'
train_csv = pandas.read_csv(csv_dir + 'all.csv',
                                    names=['image', 'filename', 'row', 'col', 'label', 'is_original'],
                                    header=0)
train_csv=train_csv.groupby(['image'])

image_sets = [
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

imgdir = '/home/ubuntu/data/mcnn_data/src/'
labeldir='/home/ubuntu/data/mcnn_data/label/'
outputdir='/home/ubuntu/data/mcnn_data/vaihingen_final/unet_label2/'
img_h2=72
img_w2=72

for i in range(len(image_sets)):
    # load test image
    imgpath = imgdir + image_sets[i]
    rgb_img = cv2.imread(imgpath)
    labelpath=labeldir+image_sets[i]
    label_img=cv2.imread(labelpath)
    print("processing " + imgpath)

    rows, cols, _ = rgb_img.shape

    seg = DatasetGenerator.segmentation_quickshift(image_path=imgpath)
    seg = seg + 1
    props = regionprops(seg)

    csv=train_csv.get_group(image_sets[i])
    filenames = csv.filename.tolist()
    rows = csv.row.tolist()
    cols = csv.col.tolist()
    is_originals=csv.is_original.tolist()

    gt=None
    for j in range(len(filenames)):

        if is_originals[j]==0:
            cv2.imwrite(outputdir + filenames[j], gt)
            continue

        row_j=rows[j]
        col_j = cols[j]

        roi_label = DatasetGenerator.create_singlescale_roi(label_img, seg, props, row_j, col_j, scale=2)
        row,col,_=roi_label.shape
        # temp = np.zeros((row, col), 'uint8')
        temp = np.zeros((row, col,3), 'uint8')
        for r in range(row):
            for c in range(col):
                # label = DatasetGenerator.bgr2label(roi_label[r, c, :], dataset_type='vaihingen')
                # temp[r, c] = label

                temp[r,c,:]=roi_label[r,c,:]

        gt = cv2.resize(temp, (img_h2, img_w2))
        cv2.imwrite(outputdir+filenames[j],gt)


