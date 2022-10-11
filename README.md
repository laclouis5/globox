# Globox — Object Detection Toolbox

This framework can:

* parse all kinds of object detection datasets (ImageNet, COCO, YOLO, PascalVOC, OpenImage, CVAT, LabelMe, etc.) and show statistics,
* convert them to other formats (ImageNet, COCO, YOLO, PascalVOC, OpenImage, CVAT, LabelMe, etc.),
* and evaluate predictions using standard object detection metrics such as AP@[.5:.05:.95], AP@50, mAP, AR<sub>1</sub>, AR<sub>10</sub>, AR<sub>100</sub>.

This framework can be used both as a library in your own code and as a command line tool. This tool is designed to be simple to use, fast and correct.

# Quick Start

## Install

You can install the package using pip:

```shell
pip install globox
```

## Use as a library

### Parse Annotations

The library has three main components:
- `BoundingBox`: represents a bounding box with a label and an optional confidence score
- `Annotation`: represent the bounding boxes annotations for one image
- `AnnotationSet`: represents annotations for a set of images (a database)

The `AnnotationSet` class contains static methods to read different databases:

```python
# COCO
coco_gts = AnnotationSet.from_coco(file_path="path/to/file.json")

# Pascal VOC
xml_gts = AnnotationSet.from_xml(folder="path/to/files/")

# YOLO
yolo_preds = AnnotationSet.from_yolo(
    folder="path/to/files/",
    image_folder="path/to/images/")
```

`Annotation` offers file-level granularity for compatible datasets:

```python
annotation = Annotation.from_labelme(file_path="path/to/file.xml")
```

For more specific implementations the `BoundingBox` class contains lots of utilities to parse bounding boxes in different formats, like the `create()` method.

`AnnotationsSets` can be combined and annotations can be added:

```python
gts = coco_gts + xml_gts
gts.add(annotation)
```

### Inspect Databases

Iterators and efficient `image_id` lookup are easy to use:

```python
if annotation in gts:
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

Datasets can be converted to and savde in other formats easily:

```python
# To Pascal VOC
coco_gts.save_xml(save_dir="pascalVOC_db/")

# TO CVAT
coco_gts.save_cvat(path="train.xml")

# To YOLO
coco_gts.save_yolo(
    save_dir="yolo_train/", 
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
```

which outputs:

```
                              COCO Evaluation
┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳...┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Label     ┃ AP 50:95 ┃   AP 50 ┃   ┃    AR S ┃    AR M ┃    AR L ┃
┡━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇...╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ airplane  │   22.72% │  25.25% │   │    nan% │  90.00% │   0.00% │
│ apple     │   46.40% │  57.43% │   │  48.57% │    nan% │    nan% │
│ backpack  │   54.82% │  85.15% │   │ 100.00% │  72.00% │   0.00% │
│ banana    │   73.65% │  96.41% │   │    nan% │ 100.00% │  70.00% │
.           .          .         .   .         .         .         .
.           .          .         .   .         .         .         .
.           .          .         .   .         .         .         .
├───────────┼──────────┼─────────┼...┼─────────┼─────────┼─────────┤
│ Total     │   50.36% │  69.70% │   │  65.48% │  60.31% │  55.37% │
└───────────┴──────────┴─────────┴...┴─────────┴─────────┴─────────┘
```

The array of results can be saved in CSV format:

```python
evaluator.save_csv("where/to/save/results.csv")
```

Custom evaluations can be achieved with:

```python
evaluation = evaluator.evaluate(
    iou_threshold=0.33,
    max_detections=1_000,
    size_range=(0.0, 10_000))

ap = evaluation.ap()
cat_ar = evaluation["cat"].ar
```

Evaluations are cached by `(iou_threshold, max_detections, size_range)` keys. This means that you should not care about about performance, repetead queries to the evaluator are fast!

## Use in command line

Get a summary of annotations for one dataset:

```shell
globox summary /yolo/folder/ --format yolo
```

Convert annotations from one format to another one:

```shell
globox convert input/yolo/folder/ output_coco_file_path.json --format yolo --save_fmt coco
```

Evaluate a set of detections with COCO metrics:

```shell
globox evaluate groundtruths/ predictions.json --format yolo --format_dets coco
```

Show the help message for an exhaustive list of options:

```shell
globox summary -h
globox convert -h
globox evaluate -h
```

--- 

## Tests

1. Clone the repo with its test data:

```shell
git clone https://github.com/laclouis5/globox --recurse-submodules=tests/globox_test_data
```

2. Install developement dependencies (virtual env recommended):

```shell
pip install -e ".[dev]"
```

3. Run tox:

```shell
tox
```

## Speed

<details>
<summary>Click to expand</summary>

Speed benchmark can be executed with:

```shell
python3 tests/benchmark.py
```

Speed test is done using `timeit` with 5 iterations on an early 2015 MacBook Air (8 GB RAM Dual-Core 1.6 GHz). The dataset is COCO 2017 Validation which comprises 5k images and 36 781 bounding boxes. 

Task   |COCO |CVAT |OpenImage|LabelMe|PascalVOC|YOLO |TXT
-------|-----|-----|---------|-------|---------|-----|-----
Parsing|0.52s|0.59s|3.44s    |1.84s  |2.45s    |3.01s|2.54s
Saving |1.12s|0.74s|0.42s    |4.39s  |4.46s    |3.75s|3.52s

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
- [x] CLI interface
- [x] Make `image_size` optional and raise err when required (bbox conversion)
- [x] Make file saving atomic with a temporary to avoid file corruption
- [x] Pip package!
- [ ] PascalVOC Evaluation
- [ ] Parsers for TFRecord and TensorFlow
- [ ] UI interface?

## Acknowledgement

This repo is based on the work of [Rafael Padilla](https://github.com/rafaelpadilla/review_object_detection_metrics). The goal of this repo is to improve the performance and flexibility and to provide additional tools.

## Contribution

Feel free to contribute, any help you can offer with this project is most welcome. Some suggestions where help is needed:
* CLI tools and scripts
* Building a PIP package
* Developing a UI interface
