# Panoptic segmentation
## Mask-RCNN
### Requirements:
- PyTorch 10.0 from a nightly release. It **will not** work with 10.0 nor 10.0.1. Installation instructions can be found in https://pytorch.org/get-started/locally/. If you already have a diffent version, plpanease unistall it befores re-install.
- torchvision from master
- cocoapi
- yacs
- scikit-image
- matplotlib
- GCC >= 4.9
- OpenCV version 3.4.3
  - You can run this line to install it: ```pip install opencv-python==3.4.3.18```
- CUDA >= 9.0 (Keep in mind that in case of running on the servers, the CUDA version is 10.0)

### Mask-RCNN Installation
```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name maskrcnn_benchmark -y
conda activate maskrcnn_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython pip

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=10.0 

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
rm -rf apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR
``` 
### Suggestions
- After the installation of Mask-RCNN, you might want to clone the homework's repo (**this one**) inside Mask-RCNN's installation folder (typically called ```maskrcnn-benchmark```); and move the ```default.py``` we are giving you to ```maskrcnn-benchmark/maskrcnn_benchmark/config/```, replacing the original one. If you are getting an error related to a ```INPUT.VERTICAL_FLIP_PROB_TRAIN``` operation, this would correct it.

# Homework
1. Change the backbone of the architecture and compare the results obtained with the original code. Discuss the quality of segmentation.
2. Implement the Panoptic Quality metric.
