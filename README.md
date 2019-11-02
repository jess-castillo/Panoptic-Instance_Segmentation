# Homework

## Instance Segmentation

1. Change the model to Mask-RCNN and Faster-RCNN using the detection models from torchvision.

For do this you will need to import the following modules:

```
from torchvision.models.detection import FasterRCNN, MaskRCNN


from torchvision.models.detection.rpn import AnchorGenerator
```
The FasterRCNN model needs four things which are:

1)A backbone.

2)The number of classes (in this case 2).

3)An anchor generator for the bounding boxes.

4)A roipooler.

The MaskRCNN model need five things which are:

1)A backbone.

2)The number of classes (in this case 2).

3)An anchor generator for the bounding boxes.

4)A roipooler.

5)A maskroipooler.

2. Follow the instructions presented on the README.md under the Instance_Segmentation folder in order to install all the requirements you need. 

3. Report and discuss your qualitative results. Show the metrics (all of the AP and AR metrics) and discuss based on them.

## Panoptic segmentation
1. Change the backbone of the architecture and compare the results obtained with the original code. Discuss the quality of segmentation.

- (Optional) Implement the Panoptic Quality metric.

2. Following the steps presented on ```Panoptic_Segmentation/UPSNet``` folder, present the results of UPSNet using the COCO val2017 dataset, using pre-trained weights. Report the panoptic quality (PQ) and show some qualitative results. 








