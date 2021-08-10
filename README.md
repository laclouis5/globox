# Object Detection Evaluation Library (WIP)
Parse all kinds of object detection databases (ImageNet, COCO, YOLO, Pascal, OpenImage, CVAT, LabelMe, etc.) and evaluate predictions with standard object detection metrics (AP@[.5:.05:.95], AP@50, mAP, AR<sub>1</sub>, AR<sub>10</sub>, AR<sub>100</sub>, etc.).

## Install
Requires Python >= 3.8.2. Best to use a virtual environment.

```console
python3.8 -m venv .env
pip install -U pip
pip install -r requirements.txt
```

## Tests
Run tests with `python tests.py`.

## TODO
- [x] Basic data structures and utilities
- [x] Parsers (ImageNet, COCO, YOLO, Pascal, OpenImage, CVAT, LabelMe)
- [x] Parser tests
- [ ] Parsers for TFRecord and TensorFlow
- [ ] Evalutators
- [ ] Tests with a huge load
- [ ] Visualization options
- [ ] Database summary and stats
- [ ] CLI interface
- [ ] Pip package
- [ ] UI interface

## Acknowledgement
This repo is based on the work of [Rafael Padilla](https://github.com/rafaelpadilla/review_object_detection_metrics).