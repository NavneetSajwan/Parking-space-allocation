# Parking-space-allocation

This project is about automatically detecting whether a vehicle is parked in the parking spot or not. Also, we count the number of vacant and occupied spaces.

Since, it is a classic object detection problem, to generate a vanilla baseline solution I chose a pretrained model from Detectron2 modelzoo. The model is trained on COCO dataset which means, our model is ready to detect cars, buses and trucks.

## Approach

Detecting vehicles is no more a challenge with off-the-self object detection models. Since, cars are already present in COCO dataset we don't even need to fine tune the model to our custom classes. Once we detect the vehicle, we can measure the amount of overlap between the parking spot and the car. If the amount of overlap is above a threshold, we signal vehicle is parked in the parking spot.

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

Next thing in line is to label the parking spots. You are free to go ahead and manually label the parking spots in the image with a labelling tool of your choice. I was about to do the same just before I found that each image in the dataset is associated with a label in xml format. Each image has 39 predefined parking spots. Depending upon whether a car is parked there or not, these are labelled as vacant or occupied alongwith the coordinates of the rectangles that cover the parking spots.

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
    xs = []
    ys = []
    for item in coordinates:
      x, y = int(item['_x']), int(item['_y'])
      npts.append([x, y])
    img = draw_poly(img, npts)
  return img

```


One interesting thing to notice here is that, we have rotated rectangles here, while model does not predict rotated rectangles. So, if we are going to measure overlap between the two we need to either rotate the model predictions or straighten the parking spots. 

While rotated boxes definitely give us more information about how the vehicle is oriented and hence, would give accurate solution. But the caveat here is model does not have orientation info and thus the straight boxes. 

So, we go ahead and straighten the rotated boxes with this python script.

Next, we go ahead and detect cars with our model.

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





