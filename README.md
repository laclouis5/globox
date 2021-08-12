# Object Detection Evaluation Library
Parse all kinds of object detection databases (ImageNet, COCO, YOLO, PascalVOC, OpenImage, CVAT, LabelMe, etc.).

Save databases to other formats (ImageNet, COCO, YOLO, PascalVOC, OpenImage, CVAT, LabelMe, etc.).

Evaluate predictions with standard object detection metrics (AP@[.5:.05:.95], AP@50, mAP, AR<sub>1</sub>, AR<sub>10</sub>, AR<sub>100</sub>, etc.) (Work in Progress).

## Install
Requires Python >= 3.8.2. Best to use a virtual environment.

```shell
python3.8 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Use

### Parse Annotations
The library has three main components:
- `BoundingBox`: represents a bounding box with a label and an optional confidence score
- `Annotation`: represent the bounding boxes annotations for one image
- `AnnotationSet`: represents annotations for a set of images (a database)

The `AnnotationSet` class contains static methods to read different databases:

```python
coco_gts = AnnotationSet.from_coco(file_path="path/to/json_file.json")
xml_gts = AnnotationSet.from_xml(folder="path/to/xml_files/")  # PascalVOC
yolo_preds = AnnotationSet.from_yolo(folder="path/to/txt_files/")
```

`Annotation` offers file-level granularity for compatible datasets:

```python
one_annotation = Annotation.from_labelme(file_path="path/to/xml_file.xml")
```

For more specific implementations the `BoundingBox` class contains lots of utilities to parse bounding boxes in different formats, like the `create()` method.

`AnnotationsSets` can be combined and annotations can be added:

```python
gts = coco_gts + xml_gts
gts.add(one_annotation)
```

### Inspect Databases
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

### Convert and Save to many Formats
Datasets can be converted to other formats easily:

```python
coco_gts.save_xml(save_dir="pascalVOC_db/")
coco_gts.save_cvat(path="train.xml")
coco_gts.save_yolo(
    save_dir="yolo_train", 
    label_to_id={"cat": 0, "dog": 1, "racoon": 2})
```

### Future Directions (WIP)
Evaluation:

```python
evaluation = Evaluator.evaluate(
    references=coco_gts,
    predictions=yolo_preds)

print(evaluation.mAP)
print(evaluation["cat"].cocoAP)
```

## Tests
Run tests with `python tests.py`.

## TODO
- [x] Basic data structures and utilities
- [x] Parsers (ImageNet, COCO, YOLO, Pascal, OpenImage, CVAT, LabelMe)
- [x] Parser tests
- [x] Database summary and stats
- [x] Database converters
- [x] Visualization options
- [ ] Visualization options ++ (graphs, figures, ...)
- [ ] Parsers for TFRecord and TensorFlow
- [ ] Evalutators
- [ ] Tests with a huge load
- [ ] CLI interface
- [ ] Pip package
- [ ] UI interface

## Acknowledgement
This repo is based on the work of [Rafael Padilla](https://github.com/rafaelpadilla/review_object_detection_metrics). The goal of this repo is to improve the performance and flexibility and to provide additional tools.

## Contribution
Feel free to contribute, any help you can offer with this project is most welcome. Some suggestions where help is needed:
* CLI tools and scripts
* Building a PIP package
* Developing a UI interface
