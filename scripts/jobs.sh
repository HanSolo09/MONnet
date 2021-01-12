echo 'Start generating data...'
python ../gen_data.py \
  --seg_method "watershed" \
  --output_dir "/home/irsgis/data/MONet_data/watershed_train_data"
echo 'Generating data done!'

echo 'Start training...'
python ../train.py \
  --model_type "MONet" \
  --n_label 6 \
  --train_data_dir "/home/irsgis/data/MONet_data/watershed_train_data" \
  --output_dir "/home/irsgis/data/MONet_data/training/20210112_watershed" \
  --epochs 64 \
  --batch_size 128 \
  --cuda_visible_devices "1"
echo 'Training done!'

echo 'Start prediction and evaluation...'
python ../predict.py \
  --trained_model "/home/irsgis/data/MONet_data/training/20210112_watershed/MONet_weights.hdf5" \
  --image_dir "/home/irsgis/data/MONet_data/image" \
  --dataset_type "vaihingen" \
  --output_dir "/home/irsgis/data/MONet_data/training/20210112_watershed" \
  --seg_dir "/home/irsgis/data/MONet_data/optimizing" \
  --evaluation_path "/home/irsgis/data/MONet_data/watershed_train_data/test_list.csv" \
  --seg_method "watershed" \
  --cuda_visible_devices "1"
echo 'Prediction and evaluation done!'
