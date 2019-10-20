# Instance Segmentation
### Requirements:
- setuptools
- pycocotools

### Penn-udan Database for Pedestrian Detection and Segmentation
```
Dowload:
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip .
Extract it in the current folder:
unzip PennFudanPed.zip
```

### For training and evaluation functions:
```
# Download TorchVision repo to use some files from references/detection
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0

cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
```

