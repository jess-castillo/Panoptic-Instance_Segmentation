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
from maskrcnn_benchmark.config import cfg
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


from PIL import Image
import json
import logging
import torch
import numpy as np
import skimage.draw as draw
import tempfile
from pycocotools.coco import COCO
import os
import shutil
import sys
import random
import math
import re
import time
import cv2
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torchvision import transforms as T
from torchvision.transforms import functional as F
# Helper Functions for the Shapes Dataset

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

class ShapeDataset(object):
  
  def __init__(self, num_examples, transforms=None):
    
    self.height = 128
    self.width = 128
    
    self.num_examples = num_examples
    self.transforms = transforms # IMPORTANT, DON'T MISS
    self.image_info = []
    self.logger = logging.getLogger(__name__)
    
    # Class Names: Note that the ids start fromm 1 not 0. This repo uses the 0 index for background
    self.class_names = {"square": 1, "circle": 2, "triangle": 3}
    
    # Add images
    # Generate random specifications of images (i.e. color and
    # list of shapes sizes and locations). This is more compact than
    # actual images. Images are generated on the fly in load_image().
    for i in range(num_examples):
        bg_color, shapes = self.random_image(self.height, self.width)
        self.image_info.append({ "path":None,
                       "width": self.width, "height": self.height,
                       "bg_color": bg_color, "shapes": shapes
                       })
    
    # Fills in the self.coco varibale for evaluation.
    self.get_gt()
    
    # Variables needed for coco mAP evaluation
    self.id_to_img_map = {}
    for i, _ in enumerate(self.image_info):
      self.id_to_img_map[i] = i

    self.contiguous_category_id_to_json_id = { 0:0 ,1:1, 2:2, 3:3 }
    

  def random_shape(self, height, width):
    """Generates specifications of a random shape that lies within
    the given height and width boundaries.
    Returns a tuple of three valus:
    * The shape name (square, circle, ...)
    * Shape color: a tuple of 3 values, RGB.
    * Shape dimensions: A tuple of values that define the shape size
                        and location. Differs per shape type.
    """
    # Shape
    shape = random.choice(["square", "circle", "triangle"])
    # Color
    color = tuple([random.randint(0, 255) for _ in range(3)])
    # Center x, y
    buffer = 20
    y = random.randint(buffer, height - buffer - 1)
    x = random.randint(buffer, width - buffer - 1)
    # Size
    s = random.randint(buffer, height//4)
    return shape, color, (x, y, s)

  def random_image(self, height, width):
      """Creates random specifications of an image with multiple shapes.
      Returns the background color of the image and a list of shape
      specifications that can be used to draw the image.
      """
      # Pick random background color
      bg_color = np.array([random.randint(0, 255) for _ in range(3)])
      # Generate a few random shapes and record their
      # bounding boxes
      shapes = []
      boxes = []
      N = random.randint(1, 4)
      labels = {}
      for _ in range(N):
          shape, color, dims = self.random_shape(height, width)
          shapes.append((shape, color, dims))
          x, y, s = dims
          boxes.append([y-s, x-s, y+s, x+s])

      # Apply non-max suppression wit 0.3 threshold to avoid
      # shapes covering each other
      keep_ixs = non_max_suppression(np.array(boxes), np.arange(N), 0.3)
      shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
      
      return bg_color, shapes
  
  
  def draw_shape(self, image, shape, dims, color):
      """Draws a shape from the given specs."""
      # Get the center x, y and the size s
      x, y, s = dims
      if shape == 'square':
          cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
      elif shape == "circle":
          cv2.circle(image, (x, y), s, color, -1)
      elif shape == "triangle":
          points = np.array([[(x, y-s),
                              (x-s/math.sin(math.radians(60)), y+s),
                              (x+s/math.sin(math.radians(60)), y+s),
                              ]], dtype=np.int32)
          cv2.fillPoly(image, points, color)
      return image, [ x-s, y-s, x+s, y+s]


  def load_mask(self, image_id):
    """
    Generates instance masks for shapes of the given image ID.
    """
    info = self.image_info[image_id]
    shapes = info['shapes']
    count = len(shapes)
    mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
    boxes = []
    
    for i, (shape, _, dims) in enumerate(info['shapes']):
        mask[:, :, i:i+1], box = self.draw_shape( mask[:, :, i:i+1].copy(),
                                            shape, dims, 1)
        boxes.append(box)
    
    
    # Handle occlusions
    occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    for i in range(count-2, -1, -1):
        mask[:, :, i] = mask[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
    
    segmentation_mask = mask.copy()
    segmentation_mask = np.expand_dims(np.sum(segmentation_mask, axis=2), axis=2)
    
    # Map class names to class IDs.
    class_ids = np.array([self.class_names[s[0]] for s in shapes])
    return segmentation_mask.astype(np.uint8), mask.astype(np.uint8), class_ids.astype(np.int32), boxes
  
  def load_image(self, image_id):
    """Generate an image from the specs of the given image ID.
    Typically this function loads the image from a file, but
    in this case it generates the image on the fly from the
    specs in image_info.
    """
    info = self.image_info[image_id]
    bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
    image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
    image = image * bg_color.astype(np.uint8)
    for shape, color, dims in info['shapes']:
        image, _ = self.draw_shape(image, shape, dims, color)
    return image
      
  def __getitem__(self, idx):
    
    """Generate an image from the specs of the given image ID.
    Typically this function loads the image from a file, but
    in this case it generates the image on the fly from the
    specs in image_info.
    """
    image = Image.fromarray(self.load_image(idx))
    segmentation_mask, masks, labels, boxes = self.load_mask(idx)
    
    # create a BoxList from the boxes
    boxlist = BoxList(boxes, image.size, mode="xyxy")

    # add the labels to the boxlist
    boxlist.add_field("labels", torch.tensor(labels))

    # Add masks to the boxlist
    masks = np.transpose(masks, (2,0,1))
    masks = SegmentationMask(torch.tensor(masks), image.size, "mask")
    boxlist.add_field("masks", masks)
    
    # Add semantic segmentation masks to the boxlist for panoptic segmentation
    segmentation_mask = np.transpose(segmentation_mask, (2,0,1))
    seg_masks = SegmentationMask(torch.tensor(segmentation_mask), image.size, "mask")
    boxlist.add_field("seg_masks", seg_masks)
    
    # Important line! dont forget to add this
    if self.transforms:
        image, boxlist = self.transforms(image, boxlist)

    # return the image, the boxlist and the idx in your dataset
    return image, boxlist, idx
  
  
  def __len__(self):
      return self.num_examples
    

  def get_img_info(self, idx):
      # get img_height and img_width. This is used if
      # we want to split the batches according to the aspect ratio
      # of the image, as it can be more efficient than loading the
      # image from disk

      return {"height": self.height, "width": self.width}
    
  def get_gt(self):
      # Prepares dataset for coco eval
      
      
      images = []
      annotations = []
      results = []
      
      # Define categories
      categories = [ {"id": 1, "name": "square"}, {"id": 2, "name": "circle"}, {"id": 3, "name": "triangle"}]


      i = 1
      ann_id = 0

      for img_id, d in enumerate(self.image_info):

        images.append( {"id": img_id, 'height': self.height, 'width': self.width } )

        for (shape, color, dims) in d['shapes']:
          
          if shape == "square":
            category_id = 1
          elif shape == "circle":
            category_id = 2
          elif shape == "triangle":
            category_id = 3
          
          x, y, s = dims
          bbox = [ x - s, y - s, x+s, y +s ] 
          area = (bbox[0] - bbox[2]) * (bbox[1] - bbox[3])
          
          # Format for COCOC
          annotations.append( {
              "id": int(ann_id),
              "category_id": category_id,
              "image_id": int(img_id),
              "area" : float(area),
              "bbox": [ float(bbox[0]), float(bbox[1]), float(bbox[2]) - float(bbox[0]) + 1, float(bbox[3]) - float(bbox[1]) + 1 ], # note that the bboxes are in x, y , width, height format
              "iscrowd" : 0
          } )

          ann_id += 1

      # Save ground truth file
      
      with open("tmp_gt.json", "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories }, f)

      # Load gt for coco eval
      self.coco = COCO("tmp_gt.json") 
      

train_dt = ShapeDataset(100)
im, boxlist, idx = train_dt[0]

if not os.path.exists(os.path.join(os.getcwd(),'qualitive_results')):
    os.makedirs('qualitive_results')
    

rows = 2
cols = 2
fig = plt.figure(figsize=(8, 8))
for i in range(1, rows*cols+1):
  im, boxlist, idx = train_dt[i]
  fig.add_subplot(rows, cols, i)
  plt.imshow(im)
  
fig.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'qualitive_results','prueba.png'))


