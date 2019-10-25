# Homework

## Instance Segmentation
1. Change the model to Mask-RCNN and Faster-RCNN using the detection models from torchvision.

For making this you will need to import the following modules:

from torchvision.models.detection import FasterRCNN, MaskRCNN


from torchvision.models.detection.rpn import AnchorGenerator

These two models need an anchor-generator, a backbone and a RoIpooler. You can play with  this three things in order to obtain different results. 



2. Report and discuss your qualitative results. Show the metrics (all of the AP and AR metrics) and discuss based on them.

## Panoptic segmentation
1. Change the backbone of the architecture and compare the results obtained with the original code. Discuss the quality of segmentation.

- (Optional) Implement the Panoptic Quality metric.

2. Following the steps presented on ```Panoptic_Segmentation/UPSNet``` folder, present the results of UPSNet using the COCO val2017 dataset, using pre-trained weights. Report the panoptic quality (PQ) and show some qualitative results. 








