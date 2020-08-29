# Parking-space-allocation

This project is about automatically detecting whether a vehicle is parked in the parking spot or not. Also, we count the number of vacant and occupied spaces.

Since, it is a classic object detection problem, to generate a vanilla baseline solution I chose a pretrained model from Detectron2 modelzoo. The model is trained on COCO dataset which means, our model is ready detect cars, buses and trucks.

## Approach
Detecting vehicles is no more a challenge with off-the-self object detection models. We don't even need to fine tune the model to our custom classes. Since, cars are already present in COCO dataset.
Once we detect the vehicle, we can measure the amount of overlap between the parking spot and the car. If the amount of overlap is above a threshold, we signal vehicle is parked in the parking spot.

The challenge is to define what exactly a parking spot is ? How can the model detect a parking place ? Different parking lots have different types parking spots. Can there be a solution where we detect parking spots in any parking place and if yes what would be the features of a parking spot?

We will come to these questions later. For now, we assume that the answer is no to all of them.

So, one way is to manually label all the parking spots in the image.

## Data


There is a huge dataset (approx. 10 GB)  of parking lot images on Kaggle. But we don't need that large a dataset for our problem. 
The solution we are creating has some constraints:

1. We get to choose one parking lot
2. All the test images of the parking lot must be from one camera, whose poition does not vary with images. Basically, it cannot be a hand-held camera. It can be a cctv camera.

Here is the link to the dataset:

There are three parking lots in the dataset. I chose one of them.

Next thing in line is to label the parking spots. You are free to go ahead and manually label the parking spots of your choice in the image with a labelling tool of your choice. I was about to do the same just before I found that each image in the dataset is associated with a label in xml format. Each image has 39 predefined parking spots. Depending upon whether a car is parked there or not, these are labelled as vacant or occupied alongwith the coordinates of the rectangles that cover the parking spots.

I downloaded one of them and ran a python script to convert them to store them in a PyTorch tensor. This is how it looks.

One interesting thing to notice here is that, we have rotated rectangles here, while model does not predict rotated rectangles. So, if we are going to measure overlap between the two we need to either rotate the model predictions or straighten the parking spots. 

While rotated boxes definitely give us more information about how the vehicle is oriented and hence, would give accurate solution. But the caveat here is model does not have orientation info and thus the straight boxes. 

So, we go ahead and straighten the rotated boxes with this python script.

Next, we go ahead and detect cars with our model.

I then find how many cars are overlapping to how many boxes and to what extent. If overlap is good enough, we predict the car is parked.

And this bring us to our next bit, how to measure the amount of overlap effectively?

## Calculating overlap

As far as overlap calculation is concerned, IOU is the goto solution in deep learning world. So, what is IOU?

Intersection Over Union, abbreviated as IOU is a ....

