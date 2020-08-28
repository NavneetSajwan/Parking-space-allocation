# Parking-space-allocation

This project is about automatically detecting whether a vehicle is parked in the parking spot or not. Also, we count the number of vacant and occupied spaces.

Since, it is a classic object detection problem, to generate a vanilla baseline solution I chose a pretrained model from Detectron2 modelzoo. The model is trained on COCO dataset which means, our model is ready detect cars, buses and trucks.

## Approach
Detecting vehicles is no more a challenge with off-the-self object detection models. We don't even need to fine tune the model to our custom classes. Since, cars are already present in COCO dataset.
Once we detect the vehicle, we can measure the amount of overlap between the parking spot and the car. If the amount of overlap is above a threshold, we signal vehicle is parked in the parking spot.

The challenge is to define what exactly a parking spot is ? How can the model detect a parking place ? Different parking lots have different types parking spots. Can there be a solution where we detect parking spots in any parking place and if yes what would be the features of a parking spot?

We will come to these questions later. For now, we assume that the anwer is no to all of them.

So, one way is to manually label all the parking spots in the image.

## Data


There is a huge dataset (approx. 10 GB)  of parking lot images on Kaggle. But we don't need that large a dataset for our problem.
