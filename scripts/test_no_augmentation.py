import os
import pandas

train_data_dir = ''
num_augment = 10

# read original training data
df = pandas.read_csv(os.path.join(train_data_dir, 'train_list.csv'), names=['image', 'filename', 'row', 'col', 'label'], header=0)
train_image = df.image.tolist()
train_filename = df.filename.tolist()
train_row = df.row.tolist()
train_col = df.col.tolist()
train_label = df.label.tolist()
print("Number of train data: ", len(train_filename))

# pick training data before data augmentation
new_image = train_image[0::num_augment]
new_filename = train_filename[0::num_augment]
new_row = train_row[0::num_augment]
new_col = train_col[0::num_augment]
new_label = train_label[0::num_augment]

# save results
df_new = pandas.DataFrame(data={"image": new_image, "filename": new_filename, "row": new_row, "col": new_col, "label": new_label})
df_new.to_csv(os.path.join(train_data_dir, 'train_list_no_augmentation.csv'), sep=',', index=False)
print("Number of train data (no augmentation): ", len(new_filename))
