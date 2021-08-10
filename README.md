# Object Detection Evaluation Library (WIP)
Parse all kinds of object detection databases (ImageNet, COCO, YOLO, Pascal, OpenImage, CVAT, LabelMe, etc.) and evaluate predictions with standard object detection metrics.

## Install
Requires Python >= 3.8.2. Best to use a virtual environment.

```shell
python3.8 -m venv .env
pip install -U pip
pip install -r requirements.txt
```

## Tests
Run tests with `python tests.py`.

## Aknowledgement
This repo is based on the work of [Rafael Padilla](https://github.com/rafaelpadilla/review_object_detection_metrics).