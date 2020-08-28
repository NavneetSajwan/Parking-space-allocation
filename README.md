# Parking-space-allocation

This project is about automatically detecting whether a vehicle is parked in the parking spot or not. Also, we count the number of vacant and occupied spaces.

Since, it is a classic object detection problem, to generate a vanilla baseline solution I chose a pretrained model from Detectron2 modelzoo. The model is trained on COCO dataset which means, our model is ready detect cars, buses and trucks.

## Data

Before our engines roar, we need to get the fuel for them.
There is a huge dataset (approx. 10 GB)  of parking lot images on Kaggle. But we don't need all that, since we have already have pretrained model.
