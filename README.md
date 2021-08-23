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

### COCO Evaluation

Evaluating is as easy as:

```python
evaluator = COCOEvaluator(coco_gts, yolo_preds)
ap = evaluator.ap()
```

All COCO metrics are available:

```python
ar_100 = evaluator.ar_100()
ap_75 = evaluator.ap_75()
ap_small = evaluator.ap_small()
...
```

All COCO standard metrics can be displayed in a pretty printed table with:

```python
evaluator.show_summary()
````

which outputs:

```bash
  COCO Evaluation  
┏━━━━━━━━┳━━━━━━━━┓
┃ Metric ┃  Value ┃
┡━━━━━━━━╇━━━━━━━━┩
│ AP     │ 50.36% │
│ AP 50  │ 69.70% │
│ AP 75  │ 57.17% │
│ AP S   │ 59.33% │
│ AP M   │ 55.80% │
│ AP L   │ 48.94% │
│ AR 1   │ 38.68% │
│ AR 10  │ 59.37% │
│ AR 100 │ 59.54% │
│ AR S   │ 65.48% │
│ AR M   │ 60.31% │
│ AR L   │ 55.37% │
└────────┴────────┘
```
This can be slow because it requires 90 iterations of the COCO evaluation metric on the validation set, thus a progress bar is shown.

Custom evaluation policy can be done with:

```python
evaluation = evaluator.evaluate(
    iou_threshold=0.33,
    max_detections=1_000,
    size_range=(0.0, 10_000))

print(evaluation.ap(), evaluation.ar())
```

Evaluations are cached by `(iou_threshold, max_detections, size_range)` key for performance reasons. This avoids re-computations, for instance querying `.ap_50()` after `.ap()` has been called will not incur a re-computation. When dealing with large datasets the cache can grow very large, thus `.clear_cache()` method can be called to empty it.

Evaluations can be queried by class label:

```python
key = (0.5, 100, None)  # AP_50
cat_eval = evaluator.evaluations[key]["cat"]
cat_ap = cat_eval.ap()
```

## Tests

Run tests with `python tests.py`.

## Speed

<details>
<summary>Click to expand</summary>

Speed test is done using `timeit` with 1 iteration on an early 2015 MacBook Air (8 GB RAM Dual-Core 1.6 GHz). The database is COCO 2017 Validation which comprises 5k images and 36 781 bounding boxes. 

Task|COCO|CVAT|OpenImage|LabelMe|PascalVOC|YOLO|TXT
----|----|----|---------|-------|---------|----|---
Parsing|0.34s|0.84s|24.32s|9.64s|4.12s|20.55s|20.55s
Saving |0.33s|0.71s|0.44s |4.53s|4.30s|2.50s |2.31s

OpenImage, YOLO and TXT are slower because they store bounding box coordinates in relative coordinates and do not provide the image size, so reading it from the image file is required.

The fastest format is COCO and LabelMe (for individual annotation files).

`AnnotationSet.show_stats()`: 0.12 s

</details>

## TODO

- [x] Basic data structures and utilities
- [x] Parsers (ImageNet, COCO, YOLO, Pascal, OpenImage, CVAT, LabelMe)
- [x] Parser tests
- [x] Database summary and stats
- [x] Database converters
- [x] Visualization options
- [x] COCO Evaluation
- [x] Tests with a huge load (5k images)
- [ ] Visualization options ++ (graphs, figures, ...)
- [ ] Parsers for TFRecord and TensorFlow
- [ ] PascalVOC Evaluation
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
