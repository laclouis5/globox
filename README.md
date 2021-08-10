# Object Detection Evaluation Library (WIP)
Parse all kinds of object detection databases (ImageNet, COCO, YOLO, Pascal, OpenImage, CVAT, LabelMe, etc.) and evaluate predictions with standard object detection metrics (AP@[.5:.05:.95], AP@50, mAP, AR<sub>1</sub>, AR<sub>10</sub>, AR<sub>100</sub>, etc.).

## Install
Requires Python >= 3.8.2. Best to use a virtual environment.

```console
python3.8 -m venv .env
pip install -U pip
pip install -r requirements.txt
```

## Use (WIP)
There are three main components:
- `BoundingBox`: represents a bounding box with a label and an optional confidence score
- `Annotation`: represent the bounding boxes annotations for one image
- `AnnotationSet`: represents annotations for a set of images (a database)

The `AnnotationSet` class contains static methods to read different databases:

```python
coco_gts = AnnotationSet.from_coco(file_path: "path/to/json_file.json")
yolo_preds = AnnotationSet.from_yolo(folder: "path/to/txt_files/")
```

However, `Annotation` offers file granularity for compatible datasets:

```python
annotation = Annotation.from_labelme(file_path: "path/to/xml_file.xml")
```

For more specific implementations the `BoundingBox` class contains lots of utilities to parse bounding boxes.

---
### Preview (WIP)
Evaluation will be performed via:

```python
evaluation = Evaluator.evaluate(
    references = coco_gts,
    predictions = yolo_preds)

print(evaluation.mAP)
print(evaluation["cat"].cocoAP)
```

Database stats will be printable:

```python
coco_gts.show_stats()
```

Databases will be convertible:

```python
coco_gts.to_yolo(save_dir: "path/to/data/")
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
- [ ] Database converters

## Acknowledgement
This repo is based on the work of [Rafael Padilla](https://github.com/rafaelpadilla/review_object_detection_metrics). The goal of this repo is to improve the performance and flexibility.

## Contribution
Feel free to contribute, any help you can offer with this project is most welcome. Some suggestions where help is needed:
* CLI tools and scripts
* Building a PIP package
* Developing a UI interface
