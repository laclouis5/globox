from globox import *
from pathlib import Path
from timeit import timeit
from time import perf_counter
from rich.table import Table
from rich import print as rich_print
from argparse import ArgumentParser
from tempfile import TemporaryDirectory

import constants as cst


def benchmark(repetitions: int = 5):
    base = (Path(__file__).parent / "globox_test_data/coco_val_5k/").resolve()
    coco = base / "coco.json"
    images = base / "images"

    gts = AnnotationSet.from_coco(coco)
    labels = sorted(gts._labels())
    label_to_id = {l: i for i, l in enumerate(labels)}

    coco_gt = AnnotationSet.from_coco(cst.coco_gts_path)
    coco_det = coco_gt.from_results(cst.coco_results_path)

    evaluator = COCOEvaluator(
        ground_truths=coco_gt, 
        predictions=coco_det,
    )

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        
        coco_out = tmp / "coco_out.json"
        cvat = tmp / "cvat.xml"
        oi = tmp / "openimage.csv"
        labelme = tmp / "labelme"
        xml = tmp / "xml"
        yolo = tmp / "yolo"
        txt = tmp / "txt"
        
        start = perf_counter()

        coco_s = timeit(lambda: gts.save_coco(coco_out), number=repetitions)
        cvat_s = timeit(lambda: gts.save_cvat(cvat), number=repetitions)
        oi_s = timeit(lambda: gts.save_openimage(oi), number=repetitions)
        labelme_s = timeit(lambda: gts.save_labelme(labelme), number=repetitions)
        xml_s = timeit(lambda: gts.save_xml(xml), number=repetitions)
        yolo_s = timeit(lambda: gts.save_yolo_darknet(yolo, label_to_id=label_to_id), number=repetitions)
        txt_s = timeit(lambda: gts.save_txt(txt, label_to_id=label_to_id), number=repetitions)

        coco_p = timeit(lambda: AnnotationSet.from_coco(coco), number=repetitions)
        cvat_p = timeit(lambda: AnnotationSet.from_cvat(cvat), number=repetitions)
        oi_p = timeit(lambda: AnnotationSet.from_openimage(oi, image_folder=images), number=repetitions)
        labelme_p = timeit(lambda: AnnotationSet.from_labelme(labelme), number=repetitions)
        xml_p = timeit(lambda: AnnotationSet.from_xml(xml), number=repetitions)
        yolo_p = timeit(lambda: AnnotationSet.from_yolo_darknet(yolo, image_folder=images), number=repetitions)
        txt_p = timeit(lambda: AnnotationSet.from_txt(txt, image_folder=images), number=repetitions)

    eval_t = timeit(lambda: evaluator._evaluate_all(), number=repetitions) / repetitions
    stats_t = timeit(lambda: gts.show_stats(), number=repetitions) / repetitions

    stop = perf_counter()

    headers = ["COCO", "CVAT", "Open Image", "LabelMe", "Pascal VOC", "YOLO", "Txt"]
    
    parse_times = (
        f"{(t / repetitions):.2f}s" 
        for t in (coco_p, cvat_p, oi_p, labelme_p, xml_p, yolo_p, txt_p)
    )
    save_times = (
        f"{(t / repetitions):.2f}s" 
        for t in (coco_s, cvat_s, oi_s, labelme_s, xml_s, yolo_s, txt_s)
    )

    table = Table(title=f"Benchmark ({len(gts)} images, {gts.nb_boxes()} boxes)")
    table.add_column("")
    for header in headers:
        table.add_column(header, justify="right")
    
    table.add_row("Parsing", *parse_times)
    table.add_row("Saving", *save_times)

    rich_print(table)

    print(f"Show stats time: {stats_t:.2f} s")
    print(f"Evaluation time: {eval_t:.2f} s")
    print(f"Total benchmark duration: {(stop - start):.2f} s")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repetitions", "-n", default=5, type=int, 
        help="Number of repetitions for timeit.")
    args = parser.parse_args()
    benchmark(repetitions=args.repetitions)
