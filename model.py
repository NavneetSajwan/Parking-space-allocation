# from appp import app
import detectron2
import os
from detectron2.utils.logger import setup_logger
setup_logger()
# import tensorflow
# import some common libraries
import numpy as np
import cv2
import random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.catalog import Metadata
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode
from helper import *
import torch
import cv2
# from google.colab.patches import cv2_imshow
import numpy as np
import json 
import codecs
from torch import tensor
# label_path = 'via_project_26Aug2020_11h59m_json (1).json'



def setup_model():
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
	cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5 # lower value decreases the number of reduntant boxes
	cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
	# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
	# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
	predictor = DefaultPredictor(cfg)
	return predictor, cfg

def gen_car_bboxes(im, predictor):
  outputs = predictor(im)

  a = outputs["instances"].pred_classes
  indices = (a==2).nonzero().flatten()
  output_cars = custom_output(outputs, indices, im)

  return output_cars

def visualize_preds(outputs, cfg,  im):
  v = Visualizer(im[:, :, ::-1],
             MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
             scale=1.0,
             instance_mode = ColorMode.SEGMENTATION
             )
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  img_out = v.get_image()[:, :, ::-1]
  return img_out

def gen_bbox_predictions(im, predictor):
  outputs = predictor(im)

  a = outputs["instances"].pred_classes
  indices = (a==2).nonzero().flatten()
  op = custom_output(outputs, indices, im)

  preds = op['instances'].pred_boxes
  torch_preds = preds.tensor
  torchint_preds = torch_preds.type(torch.IntTensor)
  print('tip type', type(torchint_preds))
  return torchint_preds

def draw_output(torchint_preds, torch_bbox, img, iou_threshold = 0.3):
  #loop over all the predefined parking spot boxes
  for label in torch_bbox:
      iou_list = []
      #loop over model predicted boxes
      for pred in torchint_preds:
        iou = bb_intersection_over_union(label, pred)
        iou_list.append(iou)
        iou_max = max(iou_list)
        if iou_max > iou_threshold:
          #if spot is occupied, draw a green box
          color = (0,255,0)
        else:
          #if spot is empty, draw a red box
          color = (0,0,255)
      img = cv2.rectangle(img,(label[0], label[1]), (label[2], label[3]), color, 2)
  return img

# def draw_output(torchint_preds, torch_bbox, img):
# 	for label in torch_bbox:
# 	    iou_list = []
# 	    for pred in torchint_preds:
# 	      iou = bb_intersection_over_union(label, pred)
# 	      iou_list.append(iou)
# 	      iou_max = max(iou_list)
# 	      if iou_max > 0.3:
# 	        color = (0,255,0)
# 	      else:
# 	        color = (0,0,255)
# 	    img = cv2.rectangle(img,(label[0], label[1]), (label[2], label[3]), color, 2)
# 	return img


# def generate_label_bboxes():
#   data = json.load(codecs.open(label_path, 'r', 'utf-8-sig')) 
#   parking_spaces = data['parking']['space']
#   bbox_arr = []
#   bbox = []
#   for parking_space in parking_spaces:
#     coordinates = parking_space['contour']['point']
#     npts = []
#     xs = []
#     ys = []
#     for item in coordinates:
#       x, y = int(item['_x']), int(item['_y'])
#       xs.append(x)
#       ys.append(y)

#     xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
#     bbox.append([xmin, ymin, xmax, ymax])
#   bbox_arr = np.asarray(bbox)
#   torch_bbox = tensor(bbox_arr)
#   return torch_bbox


def generate_label_bboxes(label_path):
  data = json.load(codecs.open(label_path, 'r', 'utf-8-sig')) 
  parking_spaces = data['parking']['space']
  bbox_arr = []
  bbox = []
  for parking_space in parking_spaces:
    coordinates = parking_space['contour']['point']
    xs = []
    ys = []
    for item in coordinates:
      x, y = int(item['_x']), int(item['_y'])
      xs.append(x)
      ys.append(y)
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    bbox.append([xmin, ymin, xmax, ymax])
  bbox_arr = np.asarray(bbox)
  torch_bbox = tensor(bbox_arr)
  return torch_bbox


def generate_label_bboxes_via(label_path):
  f = open(label_path)
  data = json.load(f)
  space_list = data['output.jpg1428438']['regions']
  bbox_list = []
  for space in space_list:
    bbox = space['shape_attributes']
    h,w,x,y = bbox['height'], bbox['width'], bbox['x'], bbox['y'] 
    x1,y1,x2,y2 = map(int,(x/3, y/3, (x + w)/3, (y + h)/3))
    bbox_list.append([x1, y1, x2, y2])
  bbox_arr = np.asarray(bbox_list)
  torch_bbox = tensor(bbox_arr)
  return torch_bbox