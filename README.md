# Parking-space-allocation

This project is about automatically detecting whether a vehicle is parked in the parking spot or not. Also, we count the number of vacant and occupied spaces.

Since, it is a classic object detection problem, to generate a vanilla baseline solution I chose a pretrained model from Detectron2 modelzoo. The model is trained on COCO dataset which means, our model is ready to detect cars, buses and trucks.

## Approach

Detecting vehicles is no more a challenge especially with off-the-self object detection models. Since, cars are already present in COCO dataset we don't even need to fine tune the model to our custom classes. Once we detect the vehicle, we can measure the amount of overlap between the parking spot and the car. If the amount of overlap is above a threshold, we signal vehicle is parked in the parking spot.

The challenge is to define what exactly a parking spot is ? How can the model detect a parking place ? Different parking lots have different types parking spots. Can there be a solution where we detect parking spots in any parking place and if yes what would be the features of a parking spot?

We will come to these questions later.

For now, we are going to manually label all the parking spots in the image.

## Data

There is a huge dataset (approx. 10 GB)  of parking lot images on Kaggle. But we don't need that large a dataset for our problem. 
The baseline solution we are creating has some constraints:

1. We get to choose one parking lot.
2. All the test images of the parking lot must be from one camera, whose position does not vary with time. Basically, it cannot be a hand-held camera e.g. cctv cameras.

Here is the link to the dataset: [Parking Lot dataset](https://www.kaggle.com/blanderbuss/parking-lot-dataset)

There are three parking lots in the dataset. I chose one of them.

Next thing in line is to label the parking spots. You can go ahead and manually label the parking spots in the image with a labelling tool of your choice. I was about to do the same just before I found that each image in the dataset is associated with a label in xml format. Each image has 39 predefined parking spots. Depending upon whether a car is parked there or not, these are labelled as vacant or occupied alongwith the coordinates of the rectangles that cover the parking spots.

I downloaded one of the label file and ran a python script to convert them to store them in a PyTorch tensor. I am more comfortable with `json` format when it comes to label and metadata, so I converted the `xml` labels to `json`. You can do that online on [Code beautifier](https://codebeautify.org/).

Once I had the `json` labels, I the called the function below to save the `parking spot` coordinates in a PyTorch tensor `torch_bbox`  

```
def generate_label_ploygons(img_path, label_path):
  data = json.load(codecs.open(label_path, 'r', 'utf-8-sig')) 
  img = cv2.imread(img_path)
  parking_spaces = data['parking']['space']
  for parking_space in parking_spaces:
    coordinates = parking_space['contour']['point']
    npts = []
    for item in coordinates:
      x, y = int(item['_x']), int(item['_y'])
      npts.append([x, y])
    img = draw_poly(img, npts)
  return img

```

![alt text](https://github.com/NavneetSajwan/Parking-space-allocation/blob/master/images/download%20(1).png "Logo Title Text 1")


One interesting thing to notice here is that, we have an irregular quadrilateral here, whereas object detection models predict rectanglular boxes. So, if we are going to calculate overlap between the two we need to turn the parking spots into rectangles. 

We have to modify the above code slightly, to get rectangular labels/parking spaces

```
def generate_label_bboxes(img_path, label_path):
  img = cv2.imread(img_path)
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
    img = cv2.rectangle(img,(xmin, ymin), (xmax, ymax),(0,255, 0), 2)
  return img
```
Here is how it looks

![alt text](https://github.com/NavneetSajwan/Parking-space-allocation/blob/master/images/regular_boxes.png "Logo Title Text 1")

Transformation looks great, but we need to store these boxes into an object and instead of returning the image we need to return the object

```
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
```
It now returns a Pytorch tensor `torch_bbox` containing all the `parking space boxes`

Next, we go ahead and detect cars with our model.

## Model

We use FAIR's (Facebook AI Research) Detectron2 api for our problem. It has a number of pretrained models. I am using `Faster RCNN` architecture in this project.
Using the function below first we load the model

```
def setup_model():
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
	cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5 # lower value decreases the number of reduntant boxes
	cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
	predictor = DefaultPredictor(cfg)
	return predictor, cfg
```
It returns a predictor to which we can pass numpy images as argument.

Let's write a function to visualize the output of predictor
```
def visualize_preds(outputs):
  v = Visualizer(im[:, :, ::-1],
             MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
             scale=1.0,
             instance_mode = ColorMode.SEGMENTATION
             )
  v = v.draw_instance_predictions(op["instances"].to("cpu"))
  img_out = v.get_image()[:, :, ::-1]
  return img_out
```
The above code is written around Detectron2 api. It returns a numpy image with boxes drawn around objects.
We call this function and display the image

```
img = cv2.imread(img_path)
predictor, cfg = setup_model()
outputs = predictor(img)
img_out = visualize_preds(outputs)
cv2.imshow(img_out)
```
#output detects things other than vehicles

#So, we write code to choose only the vehicles
```
def gen_car_bboxes(im, predictor):
  outputs = predictor(im)

  a = outputs["instances"].pred_classes
  indices = (a==2).nonzero().flatten()
  output_cars = custom_output(outputs, indices, im)

  return output_cars

```

#display results. works fine

#Now write a function to return boxes as pytorch tensors
```
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
```


I then find how many cars are overlapping to how many boxes and to what extent. If overlap is good enough, we predict the car is parked.

And this bring us to our next bit, how to measure the amount of overlap effectively?

## Calculating overlap

As far as overlap calculation is concerned, IOU is the goto solution in deep learning world. So, what is IOU?

Intersection Over Union, abbreviated as IOU is a ....

I found this code on PyImagesearch website to calculate IOU.

IOU threshold is hyperparameter to tune here, for this parking lot approx 0.3 worked out well.

IOU function takes in two boxes (a parking spot and a model predicted box) and gives iou of them. If a parking spot has greater iou than the threshold for any of the predicted boxes, we draw it on the image with green color otherwise with the red color.

Basically for every parking spot we loop over every predicted box and store the IOUs of each parking spot in a list. Choose the maximum iou, and if iou> threshold, voila.., spot is occupied. It is inefficient, but we will worry about that later. For now, I can just say, I stored boxes as tensors for a reason.


## Visualising results





