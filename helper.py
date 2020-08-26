
import cv2
from torch import tensor
from detectron2.structures import Instances

import cv2
# from google.colab.patches import cv2_imshow
import numpy as np
import json 
import codecs

label_path = '/content/drive/My Drive/Projects/Parking allocation/label.json'

def car_counter(image, text):
  font = cv2.FONT_HERSHEY_SIMPLEX 
  
  # Position of the text on image, It's the bottom left corner
  org = (20, 30) 
  
  # fontScale 
  fontScale = 0.75
    
  # Blue color in BGR 
  color = (255,0, 0) 
    
  # Line thickness of 2 px 
  thickness = 2

  # Using cv2.putText() method 
  image = cv2.putText(image, text , org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 
  return image

def custom_output(outputs, indices, im):
  idx  = indices
  image_height = im.shape[0]
  image_width = im.shape[1]
  # idx = (outputs["instances"].scores>thresh).nonzero().flatten()
  op_dict = {'pred_boxes':outputs["instances"].pred_boxes[idx],
           'scores':outputs["instances"].scores[idx],
           'pred_classes':outputs["instances"].pred_classes[idx]}
  op = Instances((image_height, image_width), **op_dict)
  final_op = {'instances': op}
  return final_op

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

