# MONet
## Abstract
Recently, deep learning has become the most innovative trend for a variety of high-spatial-resolution remote sensing imaging applications. However, large-scale land cover classification via traditional convolutional neural networks (CNNs) with sliding windows is computationally expensive and produces coarse results. Additionally, although such supervised learning approaches have performed well, collecting and annotating datasets for every task is extremely laborious, especially for those fully supervised cases where the pixel-level ground-truth labels are dense. In this work, we propose a new object-oriented deep learning framework that leverages residual networks with different depths to learn adjacent feature representations by embedding a multibranch architecture in the deep learning pipeline. The idea is to exploit limited training data at different neighboring scales to make a tradeoff between weak semantics and strong feature representations for operational land cover mapping tasks. We draw from established geographic object-based image analysis (GEOBIA) as an auxiliary module to reduce the computational burden of spatial reasoning and optimize the classification boundaries. We evaluated the proposed approach on two subdecimeter-resolution datasets involving both urban and rural landscapes and show that it significantly outperforms standard object-based deep learning methods and achieves an excellent inference time.

## Installation

There are some dependencies required by this project:

- Keras 2.3.1
- OpenCV 3.4.2
- pandas 1.1.5
- skimage 0.17.2
- matplotlib 3.3.3
- sklearn 0.24.0
- pydensecrf 1.0

To install them, simply run:

```bash
# create virtual environment
conda create -n monet  python=3.6
conda activate monet

# install keras
conda install tensorflow-gpu cudatoolkit=10.0
conda install keras

# install other dependencies
pip install opencv-python pandas scikit-image scikit-learn pydensecrf
```

## Usage

1. Use `gen_data.py` to generate multiple inputs for model training. See `python gen_data.py --help` for more details. The generated training data folder should be like this:

```bash
train_data
├── 0
│   ├── 0000000.png
│   ├── 0000001.png
│   └── 0000002.png
│   └── ...
├── 1
│   ├── 0000000.png
│   ├── 0000001.png
│   └── 0000002.png
│   └── ...
├── 2
│   ├── 0000000.png
│   ├── 0000001.png
│   └── 0000002.png
│   └── ...
├── all.csv
├── test_list.csv
└── train_list.csv
```

2. Use `train.py` to train different models. See `python train.py --help` for more details. After the training procedure is finished, the model weight and learning curve will be saved under certain folder.
3. Use `predict.py` to predict the results based on pretrained weights. Post-processing and evaluation will also be performed. Please see `python predict.py --help` for more details.

## Notice

This repo is only for demonstration and  some implementations are simplified. Please *DO NOT* use this code for evaluation / benchmarking. If you need to compare your results with ours, do not hesitate to contact the author(s) for assistance. 