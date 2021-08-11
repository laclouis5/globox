# Object Detection Evaluation Library (WIP)
Parse all kinds of object detection databases (ImageNet, COCO, YOLO, Pascal, OpenImage, CVAT, LabelMe, etc.) and evaluate predictions with standard object detection metrics (AP@[.5:.05:.95], AP@50, mAP, AR<sub>1</sub>, AR<sub>10</sub>, AR<sub>100</sub>, etc.).

## Install
Requires Python >= 3.8.2. Best to use a virtual environment.

```console
python3.8 -m venv .env
pip install -U pip
pip install -r requirements.txt
```

## Use

### APIs
The library has three main components:
- `BoundingBox`: represents a bounding box with a label and an optional confidence score
- `Annotation`: represent the bounding boxes annotations for one image
- `AnnotationSet`: represents annotations for a set of images (a database)

The `AnnotationSet` class contains static methods to read different databases:

```python
coco_gts = AnnotationSet.from_coco(file_path: "path/to/json_file.json")
xml_gts = AnnotationSet.from_xml(folder: "path/to/xml_files/")  # PascalVOC
yolo_preds = AnnotationSet.from_yolo(folder: "path/to/txt_files/")
```

`Annotation` offers file-level granularity for compatible datasets:

```python
one_annotation = Annotation.from_labelme(file_path: "path/to/xml_file.xml")
```

For more specific implementations the `BoundingBox` class contains lots of utilities to parse bounding boxes in different formats, like the `create()` method.

`AnnotationsSet`s can be combined and annotations can be added:

```python
gts = coco_gts + xml_gts
gts.add(one_annotation)
```

Iterators and efficient `image_id` lookup are easy to use:

```python
if one_annotation in gts:
    print("This annotation is in the DB.")

for box in gts.all_boxes:
    print(box.label, box.area, box.is_ground_truth)

for annotation in gts:
    print(f"{annotation.image_id}: {len(annotation.boxes)} boxes")

print(gts.image_ids == yolo_preds.image_ids)
```

### Visualization

Database stats can printed to the console:

```python
coco_gts.show_stats()
```

```
         Database Stats         
┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ Label       ┃ Images ┃ Boxes ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ aeroplane   │     10 │    15 │
│ bicycle     │      7 │    14 │
│ bird        │      4 │     6 │
│ boat        │      7 │    11 │
│ bottle      │      9 │    13 │
│ bus         │      5 │     6 │
│ car         │      6 │    14 │
│ cat         │      4 │     5 │
│ chair       │      9 │    15 │
│ cow         │      6 │    14 │
│ diningtable │      7 │     7 │
│ dog         │      6 │     8 │
│ horse       │      7 │     7 │
│ motorbike   │      3 │     5 │
│ person      │     41 │    91 │
│ pottedplant │      6 │     7 │
│ sheep       │      4 │    10 │
│ sofa        │     10 │    10 │
│ train       │      5 │     6 │
│ tvmonitor   │      8 │     9 │
├─────────────┼────────┼───────┤
│ Total       │    100 │   273 │
└─────────────┴────────┴───────┘
```

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
- [x] Database summary and stats
- [ ] Parsers for TFRecord and TensorFlow
- [ ] Evalutators
- [ ] Tests with a huge load
- [ ] Visualization options
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
