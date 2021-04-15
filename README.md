# [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
top 2% solution

## model 1

Detector: Faster RCNN

backbone: ResNext101-32d-4

## model 2

Detector: Faster RCNN

backbone: ResNet50

etc: attention

## model 3

Detector: Faster RCNN

backbone: RegNet12

## model 4

Detector: DetectoRS

backbone: ResNet50

## model 5

Detector: DetectoRS

backbone: ResNext101-32d-4

## anchor

ratio: [0.5, 1.0, 2.0] -> [0.33, 0.5, 1.0, 2.0, 3.0] 

scale: [8] -> [1, 2, 4, 8, 16]

## try

Detector: VFNet, Cascade RCNN, Cascade mask RCNN, Hybrid Task Casacde, Generalized Focal Loss

Backbone: EfficientNet, DenseNet

etc: pseudo labeling, segmentation, external data (NIH), Large box detector, small box detector

