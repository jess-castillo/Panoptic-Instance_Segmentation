# Panoptic segmentation
### Requirements:
- PyTorch 10.0 from a nightly release. It **will not** work with 10.0 nor 10.0.1. Installation instructions can be found in https://pytorch.org/get-started/locally/. If you already have a diffent version, please unistall it befores re-install.
- torchvision from master
- cocoapi
- yacs
- scikit-image
- matplotlib
- GCC >= 4.9
- OpenCV version 3.4.3
  - You can run this line to install it: ```pip install opencv-python==3.4.3.18```
- CUDA >= 9.0 (In case of running on the servers, the CUDA version is 10.0)

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
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

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


## UPSNet(Optional) 


### Requirements:

Pip install easydict

pip install git+https://github.com/cocodataset/panopticapi.git

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext inst



For implementing UPSNet we are going to use the COCOval2017 dataset. 

Once you have clone the repositorie run `init.sh` to build essential C++/CUDA modules which also download the pretained model.

### COCO datset and annotations 

Download the following files:

[COCOval2017]
(http://images.cocodataset.org/zips/val2017.zip) (1GB)

[Train/val annotations]
(http://images.cocodataset.org/annotations/annotations_trainval2017.zip) (241MB)

[Stuff train/val annotations]
(http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip) (1.1GB)

[Panoptic Train/Val annotations]
(http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) (821MB)


Once you have download all the files, create a folder named $COCO_ROOT which have  `annotations` and `images` folders under it.

Create a soft link by `ln -s $COCO_ROOT data/coco` under your `UPSNet current folder`.


If you don´t understand how does a softlink works check: 
https://www.cyberciti.biz/faq/creating-soft-link-or-symbolic-link/ 


Now you can check which files are under the new soft link runing
`$ ls -l  $COCO_ROOT data/coco`

Run `init_coco.sh` to prepare COCO dataset for UPSNet.

Run `download_weights.sh` to get trained model weights for COCO.


Finally you can test the model in the  COCO validation dataset and obtain the qualitative and quatitavie results runing: 


```shell
python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet50_coco_4gpu.yaml --weight_path model/upsnet_resnet_50_coco_90000.pth
```

You need to change the `.yaml` file which is located in the `experiments` folder depending on the GPU you are using.

Once the test phase has finished you will find a folder called `output`

Under that folder you will find the Panoptic Quality results in :

`output`/ `upsnet`/ `coco`/`upsnet_resnet50_coco_4gpu`/`val2017`/`results`/`pans_unified`/`results.json` 

At the  begining of the `results.json` file you will find the Panoptic Quality for all the classes of the COCO val dataset. At the end of it you will find the Panoptic quality for Things and Stuff classes. 


For the qualitative results you will find the images under the folder `pan` which you can find in:

`output`/ `upsnet`/ `coco`/`upsnet_resnet50_coco_4gpu`/`val2017`/`results`/`pans_unified`/`pan`






