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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torchvision import transforms as T
from torchvision.transforms import functional as F

"""
def visMask(im, seg_mask):
  m = seg_mask.instances.masks
  m = m.numpy().reshape([128,128])
  im = np.transpose(im, (2,0,1)) # 3, 128, 128
  res = im*m
  res = np.transpose(res, (1,2,0)) # 128, 128, 3 
  # plt.imshow(res)

############################################# Pretrained weight removal ######################################################
def removekey(d, listofkeys):
  r = dict(d)
  for key in listofkeys:
      print('key: {} is removed'.format(key))
      r.pop(key)
  return r
 """
logger_dir = 'log'

if logger_dir:
    mkdir(logger_dir)

logger = setup_logger("maskrcnn_benchmark", logger_dir, get_rank())
logger.info("Using {} GPUs".format(1))

config_file = "base_config.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)


# Add these for printing class names over your predictions.
COCODemo.CATEGORIES = [
    "__background",
    "square",
    "circle",
    "triangle"
]

demo = COCODemo(
    cfg, 
    min_image_size=800,
    confidence_threshold=0.7,
    convert_model=True)

base_model = demo.model  

# Removes pretrained weights from state dict
new_state_dict = removekey(base_model.state_dict(), [ 
                      "roi_heads.box.predictor.cls_score.weight", "roi_heads.box.predictor.cls_score.bias", 
                      "roi_heads.box.predictor.bbox_pred.weight", "roi_heads.box.predictor.bbox_pred.bias",
                     "roi_heads.mask.predictor.mask_fcn_logits.weight", "roi_heads.mask.predictor.mask_fcn_logits.bias"
                  ])

# Save new state dict, we will use this as our starting weights for our fine-tuned model
torch.save(new_state_dict, "base_model.pth")

"""
#### Segmantic segmentation loss:
class SegLoss(nn.Module):
  
  def __init__(self):
    super(SegLoss, self).__init__()

  def prepare_target(self, targets):
    labels = []

    for t in targets:
      t = t.get_field("seg_masks").get_mask_tensor().unsqueeze(0)
      labels.append(t)

    return cat(labels, dim=0).unsqueeze(1).to("cuda", dtype=torch.float32)

  def forward(self, mask, target):
    '''
      mask : Tensor
      target : list[Boxlist]
    '''
    
    target = self.prepare_target(target)

    loss = Fx.binary_cross_entropy_with_logits(mask, target)
    
    return loss


### Segmenter model:
def panoptic_upsampler_block(in_channels, out_channels, expansion):
  
  modules = []
  
  if expansion == 0:
    modules.append( make_conv3x3(
              in_channels,
              out_channels,
              dilation=1,
              stride=1,
              use_gn=True,
              use_relu=True,
              kaiming_init=True
          )) # no upsample
    
  for i in range(expansion):
    modules.append(make_conv3x3(
              in_channels if i == 0 else out_channels,
              out_channels,
              dilation=1,
              stride=1,
              use_gn=True,
              use_relu=True,
              kaiming_init=True
          ))
    modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
  
  return nn.Sequential(*modules)


class PanopticRCNN(nn.Module):

    def __init__(self, cfg, num_classes):
        super(PanopticRCNN, self).__init__()

       
        self.scale1_block = panoptic_upsampler_block(in_channels=cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,
                                                     out_channels=cfg.MODEL.PANOPTIC.CHANNEL_SIZE, expansion=3) # 1/32
        self.scale2_block = panoptic_upsampler_block(in_channels=cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,
                                                     out_channels=cfg.MODEL.PANOPTIC.CHANNEL_SIZE, expansion=2) # 1/16
        self.scale3_block = panoptic_upsampler_block(in_channels=cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, 
                                                     out_channels=cfg.MODEL.PANOPTIC.CHANNEL_SIZE, expansion=1) # 1/8
        self.scale4_block = panoptic_upsampler_block(in_channels=cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,
                                                     out_channels=cfg.MODEL.PANOPTIC.CHANNEL_SIZE, expansion=0) # 1/4
        
        self.num_classes = num_classes
        
        self.final_seg_mask = nn.Sequential(
          nn.Conv2d(kernel_size=1, in_channels=128, out_channels=self.num_classes),
          nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        
        
    def forward(self, features):
        #Arguments:
        #    features (list[Tensor]): feature maps gen post FPN, (N, C, H, W)
        #Returns:
        #    segmentation_mask: semantic segmentation mask
        
        
    
        
        x1 = self.scale1_block(features[3])
        
        x2 = self.scale2_block(features[2])
        
        x3 = self.scale3_block(features[1])
        
        x4 = self.scale4_block(features[0])
           
        x = x1 + x2 + x3 + x4
        
        seg_mask = self.final_seg_mask(x)
        
        return seg_mask


class PanopticModel(nn.Module):
        
    #Main class for Panoptic R-CNN. Currently supports boxes and masks.
    #It consists of three main parts:
    #- backbone
    #- rpn
    #- panoptic: ouputs semantic segmentation mask
    #- heads: takes the features + the proposals from the RPN and computes
    #    detections / masks from it.
    

    def __init__(self, cfg):
        super(PanopticModel, self).__init__()

        self.backbone = build_backbone(cfg)
        self.loss = SegLoss()
        self.training = True
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.panoptic = PanopticRCNN(cfg, num_classes=cfg.MODEL.PANOPTIC.NUM_CLASSES)
               

    def forward(self, images, targets=None):

        
        #Arguments:
        #    images (list[Tensor] or ImageList): images to be processed
        #    targets (list[BoxList]): ground-truth boxes present in the image (optional)
        #Returns:
        #    result (list[BoxList] or dict[Tensor]): the output from the model.
        #        During training, it returns a dict[Tensor] which contains the losses.
        #        During testing, it returns list[BoxList] contains additional fields
        #        like `scores`, `labels` and `mask` (for Mask R-CNN models).
        


        images = to_image_list(images)
        features = self.backbone(images.tensors)      
        seg_mask = self.panoptic(features)
        proposals, proposal_losses = self.rpn(images, features, targets)
      
       
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            segmentation_loss = self.loss(seg_mask, targets)
            
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(dict(segmentation_loss=segmentation_loss))
            
            return losses
             
        return seg_mask, result

### Build panoptic network:
def build_panoptic_network(cfg):
    return PanopticModel(cfg)
"""
###Train panoptic:
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')
    
def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train_panoptic(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.error("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        ))

def train_panoptic(cfg, local_rank, distributed, dataset):
    model = build_panoptic_network(cfg)

    device = torch.device('cuda')
    model.to(device)
    
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)     

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)


    data_loader = build_data_loader(cfg, dataset)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train_panoptic(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model

### Train panoptic driver
config_file = "shapes_config.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)

cfg.merge_from_list(['OUTPUT_DIR', 'segDir']) # The output folder where all our model checkpoints will be saved during training.
cfg.merge_from_list(['SOLVER.IMS_PER_BATCH', 25]) # Number of images to take insiade a single batch. This number depends on the size of your GPU
cfg.merge_from_list(['SOLVER.BASE_LR', 0.0001]) # The Learning Rate when training starts. Please check Detectron scaling rules to determine your learning for your GPU setup. 
cfg.merge_from_list(['SOLVER.MAX_ITER', 1000]) # The number of training iterations that will be executed during training. One iteration is given as one forward and backward pass of a mini batch of the network
cfg.merge_from_list(['SOLVER.STEPS', "(700, 800)"]) # These two numberes represent after how many iterations is the learning rate divided by 10. 
cfg.merge_from_list(['TEST.IMS_PER_BATCH', 1]) # Batch size during testing/evaluation
cfg.merge_from_list(['MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN', 2000]) # This determines how many region proposals to take in for processing into the stage after the RPN. The rule is 1000*batch_size = 4*1000 
cfg.merge_from_list(['SOLVER.CHECKPOINT_PERIOD', 100]) # After how many iterations does one want to save the model.
cfg.merge_from_list(['INPUT.MIN_SIZE_TRAIN', "(192, )"])
cfg.merge_from_list(['INPUT.MAX_SIZE_TRAIN', 192])
# Make the Output dir if one doesnt exist.
output_dir = cfg.OUTPUT_DIR
if output_dir:
    mkdir(output_dir)


# Start training.
#model = train_panoptic(cfg, local_rank=1, distributed=False, dataset=ShapeDataset(2000))
model = torch.load(os.path.join(os.getcwd(),'segDir','model_0003600.pth'))
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
if not os.path.exists(os.path.join(os.getcwd(),'qualitive_results')):
    os.makedirs('qualitive_results')
    
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

fig.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'qualitive_results','input.png'))
  
  




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
fig.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'qualitive_results','panoptic_results.png'))



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
fig.tight_layout()  
plt.savefig(os.path.join(os.getcwd(),'qualitive_results','InstanceSeg_ObjDetResults.png'))



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
fig.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'qualitive_results','semantic.png'))