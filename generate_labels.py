import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import json 
import codecs

label_path = '/content/drive/My Drive/Projects/Parking allocation/label.json'
img_path  = '/content/datasets_87490_201391_PKLot_PKLot_UFPR05_Sunny_2013-03-01_2013-03-01_18_13_01.jpg'

data = json.load(codecs.open(label_path, 'r', 'utf-8-sig')) 
parking_spaces = data['parking']['space']
bbox_arr = []
img = cv2.imread(img_path)
bbox = []
for parking_space in parking_spaces:
  coordinates = parking_space['contour']['point']
  npts = []
  xs = []
  ys = []
  for item in coordinates:
    x, y = int(item['_x']), int(item['_y'])
    xs.append(x)
    ys.append(y)

  xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
  bbox.append([xmin, ymin, xmax, ymax])
bbox_arr = np.asarray(bbox)