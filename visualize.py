import torch
from torch import nn
import torch.nn.functional as Fx
import datetime

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

from maskrcnn_benchmark.data.build import *
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.config import cfg
#from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size
from maskrcnn_benchmark.utils.comm import all_gather
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.engine.inference import compute_on_dataset, _accumulate_predictions_from_multiple_gpus
from maskrcnn_benchmark.data.datasets.evaluation.coco import coco_evaluation
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.image_list import to_image_list

from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads

import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from helper_functions import *
from dataset import *
from dataloader import *
from train import *

from PIL import Image
import json
import logging
import torch
import numpy as np
import skimage.draw as draw
import tempfile
from pycocotools.coco import COCO
import os
import sys
import random
import math
import re
import time
import cv2
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import pdb; pdb.set_trace()
model = torch.load(os.path.join(os.getcwd(),'segDir','model_0001300.pth'))
###Visualize:
# Load Trained Model
config_file = "shapes_config.yaml"

cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

# manual override some options
cfg.merge_from_list(['OUTPUT_DIR', 'segDir']) # The output folder where all our model checkpoints will be saved during training.

# update the config options with the config file
cfg.merge_from_file(config_file)

cfg.merge_from_list(['INPUT.MIN_SIZE_TRAIN', "(192, )"])
cfg.merge_from_list(['INPUT.MAX_SIZE_TRAIN', 192])

cfg.merge_from_list(['INPUT.MIN_SIZE_TEST', 192])
cfg.merge_from_list(['INPUT.MAX_SIZE_TEST', 192])
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])


vis_demo = COCODemo(
    cfg, 
    min_image_size=192,
    confidence_threshold=0.7)

# Add these for printing class names over your predictions.
COCODemo.CATEGORIES = [
    "__background",
    "square",
    "circle",
    "triangle"
]

# Load Dataset
dataset = ShapeDataset(50)

# Visualise Input Image
rows = 2
cols = 2
fig = plt.figure(figsize=(8, 8))
for i in range(1, rows*cols+1):
  img = dataset.load_image(i)
#   image = np.array(img)[:, :, [2, 1, 0]]
#   result = vis_demo.run_on_opencv_image(image)
  
  fig.add_subplot(rows, cols, i)
  plt.imshow(img)
plt.show()



# Visualise Results
rows = 2
cols = 2
fig = plt.figure(figsize=(8, 8))
for i in range(1, rows*cols+1):
  img = dataset.load_image(i)
  image = np.array(img)[:, :, [2, 1, 0]]
  result = vis_demo.run_on_opencv_image(image, panoptic="True")
  
  fig.add_subplot(rows, cols, i)
  plt.imshow(result)
plt.show()



# Visualise Results
rows = 2
cols = 2
fig = plt.figure(figsize=(8, 8))
for i in range(1, rows*cols+1):
  img = dataset.load_image(i)
  image = np.array(img)[:, :, [2, 1, 0]]
  result = vis_demo.run_on_opencv_image(image, objDet="True")
  
  fig.add_subplot(rows, cols, i)
  plt.imshow(result)
plt.show()



# Visualise Results
rows = 2
cols = 2
fig = plt.figure(figsize=(8, 8))
for i in range(1, rows*cols+1):
  img = dataset.load_image(i)
  image = np.array(img)[:, :, [2, 1, 0]]
  result = vis_demo.run_on_opencv_image(image, semantic="True")
  
  fig.add_subplot(rows, cols, i)
  plt.imshow(result)
plt.show()