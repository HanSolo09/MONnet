import pandas as pd
import os
import numpy as np
import cv2
from catboost import CatBoostClassifier, Pool
from post_processing import draw
import xgboost as xgb



from sklearn.preprocessing import OneHotEncoder
print(os.path.abspath('.'))
### read sample csv
df = pd.read_csv('xiangliu.csv')


print(df.head())
columnames = df.columns
print(columnames)


# group the df with image name
vector = np.empty((0,3))
label = np.empty((0,))
grouped= df.groupby(['image'])

for image, group in grouped:
    #read the raw image
    img = cv2.imread('/home/ubuntu/Desktop/xiangliu/data/patch2/' + image)

    # img = cv2.imread('/home/ubuntu/Desktop/gjh2.0/data/src/'+image)
    l = group['label'].values
    print(l.shape)
    cord = np.array((group['row'].values, group['col'].values))
    data = img[cord[0,:],cord[1,:]]

    vector = np.append(vector, data,axis = 0)
    label = np.append(label,l,axis = 0)

# onthotencoder
# ohe = OneHotEncoder()
# label = ohe.fit_transform(label.reshape(-1,1)).toarray()
print('training')
model = CatBoostClassifier(iterations=1000, depth = 2, learning_rate=0.1, loss_function='MultiClass',
                           logging_level='Verbose')

# model = xgb.XGBClassifier(booster='gbtree', objective='multi:softmax',
#                           learning_rate=0.1, max_depth=6,
#                           gamma = 0.1, n_estimators= 160)


model.fit(vector, label)
print('train_over')



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


    '2.tif',
    '3.tif',
    '8.tif',
    '9.tif',
    '14.tif',
    '16.tif',
    '20.tif',
    '22.tif',
    '26.tif',
    '28.tif',
    '33.tif'
]

for img in image_sets:
    #path = '/home/ubuntu/Desktop/gjh2.0/data/src/'+img
    path = '/home/ubuntu/Desktop/xiangliu/data/patch2/' + img
    print(path)
    predict_data = cv2.imread(path)
    preds_class = model.predict(predict_data.reshape(-1,3))
    class_map = preds_class.reshape(predict_data.shape[0], predict_data.shape[1])
    predict_img=draw(class_map,'xiangliu')
    #outpath='/home/ubuntu/Desktop/gjh2.0/data/vaihingen_final/xgboost/'
    outpath = '/home/ubuntu/Desktop/gjh2.0/data/xiangliu_final/xgboost/'
    imgname = img.split('.')[0]
    cv2.imwrite(outpath+imgname+'.png', predict_img)
print('over')











