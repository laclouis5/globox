# Globox — Object Detection Toolbox

This framework can:

* parse all kinds of object detection datasets (ImageNet, COCO, YOLO, PascalVOC, OpenImage, CVAT, LabelMe, etc.) and show statistics,
* convert them to other formats (ImageNet, COCO, YOLO, PascalVOC, OpenImage, CVAT, LabelMe, etc.),
* and evaluate predictions using standard object detection metrics such as $AP_{[.5:.05:.95]}$, $AP_{50}$, $mAP$, $AR_{1}$, $AR_{10}$, $AR_{100}$.

This framework can be used both as a library in your own code and as a command line tool. This tool is designed to be simple to use, fast and correct.

## Install

You can install the package using pip:

```shell
pip install globox
```

## Use as a Library

### Parse Annotations

The library has three main components:

* `BoundingBox`: represents a bounding box with a label and an optional confidence score
* `Annotation`: represent the bounding boxes annotations for one image
* `AnnotationSet`: represents annotations for a set of images (a database)

The `AnnotationSet` class contains static methods to read different dataset formats:

```python
# COCO
coco = AnnotationSet.from_coco(file_path="path/to/file.json")

# YOLOv5
yolo = AnnotationSet.from_yolo_v5(
    folder="path/to/files/",
    image_folder="path/to/images/"
)

# Pascal VOC
pascal = AnnotationSet.from_pascal_voc(folder="path/to/files/")
```

`Annotation` offers file-level granularity for compatible datasets:

```python
annotation = Annotation.from_labelme(file_path="path/to/file.xml")
```

For more specific implementations the `BoundingBox` class contains lots of utilities to parse bounding boxes in different formats, like the `create()` method.

`AnnotationsSets` are set-like objects. They can be combined and annotations can be added:

```python
gts = coco | yolo
gts.add(annotation)
```

### Inspect Datasets

Iterators and efficient lookup by `image_id`'s are easy to use:

```python
if annotation in gts:
    print("This annotation is present.")

if "image_123.jpg" in gts.image_ids:
    print("Annotation of image 'image_123.jpg' is present.")

for box in gts.all_boxes:
    print(box.label, box.area, box.is_ground_truth)

for annotation in gts:
    nb_boxes = len(annotation.boxes)
    print(f"{annotation.image_id}: {nb_boxes} boxes")
```

Datasets stats can printed to the console:

```python
coco_gts.show_stats()
```

```text
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

### Convert and Save to Many Formats

Datasets can be converted to and saved in other formats:

```python
# ImageNet
gts.save_imagenet(save_dir="pascalVOC_db/")

# YOLO Darknet
gts.save_yolo_darknet(
    save_dir="yolo_train/", 
    label_to_id={"cat": 0, "dog": 1, "racoon": 2}
)

# YOLOv5
gts.save_yolo_v5(
    save_dir="yolo_train/", 
    label_to_id={"cat": 0, "dog": 1, "racoon": 2},
)

# CVAT
gts.save_cvat(path="train.xml")
```

### COCO Evaluation

COCO Evaluation is also supported:

```python
evaluator = COCOEvaluator(
    ground_truths=gts, 
    predictions=dets
)

ap = evaluator.ap()
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

```text
                              COCO Evaluation
┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳...┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Label     ┃ AP 50:95 ┃  AP 50 ┃   ┃   AR S ┃   AR M ┃   AR L ┃
┡━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇...╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ airplane  │    22.7% │  25.2% │   │   nan% │  90.0% │   0.0% │
│ apple     │    46.4% │  57.4% │   │  48.5% │   nan% │   nan% │
│ backpack  │    54.8% │  85.1% │   │ 100.0% │  72.0% │   0.0% │
│ banana    │    73.6% │  96.4% │   │   nan% │ 100.0% │  70.0% │
.           .          .        .   .        .        .        .
.           .          .        .   .        .        .        .
.           .          .        .   .        .        .        .
├───────────┼──────────┼────────┼...┼────────┼────────┼────────┤
│ Total     │    50.3% │  69.7% │   │  65.4% │  60.3% │  55.3% │
└───────────┴──────────┴────────┴...┴────────┴────────┴────────┘
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
    size_range=(0.0, 10_000)
)

ap = evaluation.ap()
cat_ar = evaluation["cat"].ar
```

Evaluations are cached by `(iou_threshold, max_detections, size_range)` keys. This means that repetead queries to the evaluator are fast!

## Use in Command Line

If you only need to use Globox from the command line like an application, you can install the package through [pipx](https://pypa.github.io/pipx/):

```shell
pipx install globox
```

Globox will then be in your shell path and usable from anywhere.

### Usage

Get a summary of annotations for one dataset:

```shell
globox summary /yolo/folder/ --format yolo
```

Convert annotations from one format to another one:

```shell
globox convert input/yolo/folder/ output_coco_file_path.json --format yolo --save_fmt coco
```

Evaluate a set of detections with COCO metrics, display them and save them in a CSV file:

```shell
globox evaluate groundtruths/ predictions.json --format yolo --format_dets coco -s results.csv
```

Show the help message for an exhaustive list of options:

```shell
globox summary -h
globox convert -h
globox evaluate -h
```

## Run Tests

Clone the repo with its test data:

```shell
git clone https://github.com/laclouis5/globox --recurse-submodules=tests/globox_test_data
cd globox
```

Install dependencies with [uv](https://github.com/astral-sh/uv):

```shell
uv sync --dev
```

Run the tests:

```shell
uv run pytest tests
```

## Speed Banchmarks

Speed benchmark can be executed with:

```shell
uv run python tests/benchmark.py -n 5
```

The following speed test is performed using Python 3.11 and `timeit` with 5 iterations on a 2021 MacBook Pro 14" (M1 Pro 8 Cores and 16 GB of RAM). The dataset is COCO 2017 Validation which comprises 5k images and 36 781 bounding boxes.

Task   |COCO |CVAT |OpenImage|LabelMe|PascalVOC|YOLO |TXT
-------|-----|-----|---------|-------|---------|-----|-----
Parsing|0.22s|0.12s|0.44s    |0.60s  |0.97s    |1.45s|1.12s
Saving |0.32s|0.17s|0.14s    |1.06s  |1.08s    |0.91s|0.85s

* `AnnotationSet.show_stats()`: 0.02 s
* Evalaution: 0.30 s

</details>

## Todo

* [x] Basic data structures and utilities
* [x] Parsers (ImageNet, COCO, YOLO, Pascal, OpenImage, CVAT, LabelMe)
* [x] Parser tests
* [x] Database summary and stats
* [x] Database converters
* [x] Visualization options
* [x] COCO Evaluation
* [x] Tests with a huge load (5k images)
* [x] CLI interface
* [x] Make `image_size` optional and raise err when required (bbox conversion)
* [x] Make file saving atomic with a temporary to avoid file corruption
* [x] Pip package!
* [ ] PascalVOC Evaluation
* [ ] Parsers for TFRecord and TensorFlow
* [ ] UI interface?

## Acknowledgement

This repo is based on the work of [Rafael Padilla](https://github.com/rafaelpadilla/review_object_detection_metrics).

## Contribution

Feel free to contribute, any help you can offer with this project is most welcome.
