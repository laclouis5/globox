from globox import *
from pathlib import Path
from timeit import timeit
from time import perf_counter
from rich.table import Table
from rich import print as rich_print


def benchmark():
    iterations = 5
    base = Path("data/coco_val.nosync/")
    images = base / "images"

    coco = base / "coco.json"
    coco_out = base / "coco_out.json"
    cvat = base / "cvat.xml"
    oi = base / "openimage.csv"
    labelme = base / "labelme"
    xml = base / "xml"
    yolo = base / "yolo"
    txt = base / "txt"

    gts = AnnotationSet.from_coco(coco)
    labels = gts._labels()
    label_to_id = {str(l): i for i, l in enumerate(labels)}
    
    start = perf_counter()

    coco_s = timeit(lambda: gts.save_coco(coco_out), number=iterations) / iterations
    cvat_s = timeit(lambda: gts.save_cvat(cvat), number=iterations) / iterations
    oi_s = timeit(lambda: gts.save_openimage(oi), number=iterations) / iterations
    labelme_s = timeit(lambda: gts.save_labelme(labelme), number=iterations) / iterations
    xml_s = timeit(lambda: gts.save_xml(xml), number=iterations) / iterations
    yolo_s = timeit(lambda: gts.save_yolo(yolo, label_to_id=label_to_id), number=iterations) / iterations
    txt_s = timeit(lambda: gts.save_txt(txt, label_to_id=label_to_id), number=iterations) / iterations

    coco_p = timeit(lambda: AnnotationSet.from_coco(coco), number=iterations) / iterations
    cvat_p = timeit(lambda: AnnotationSet.from_cvat(cvat), number=iterations) / iterations
    oi_p = timeit(lambda: AnnotationSet.from_openimage(oi, image_folder=images), number=iterations) / iterations
    labelme_p = timeit(lambda: AnnotationSet.from_labelme(labelme), number=iterations) / iterations
    xml_p = timeit(lambda: AnnotationSet.from_xml(xml), number=iterations) / iterations
    yolo_p = timeit(lambda: AnnotationSet.from_yolo(yolo, image_folder=images), number=iterations) / iterations
    txt_p = timeit(lambda: AnnotationSet.from_txt(txt, image_folder=images), number=iterations) / iterations

    stats_t = timeit(lambda: gts.show_stats(), number=iterations) / iterations

    stop = perf_counter()

    headers = ["COCO", "CVAT", "Open Image", "LabelMe", "Pascal VOC", "YOLO", "Txt"]
    parse_times = (f"{t:.2f} s" for t in (coco_p, cvat_p, oi_p, labelme_p, xml_p, yolo_p, txt_p))
    save_times = (f"{t:.2f} s" for t in (coco_s, cvat_s, oi_s, labelme_s, xml_s, yolo_s, txt_s))

    table = Table(title=f"Benchmark ({len(gts)} images, {gts.nb_boxes()} boxes)")
    table.add_column("")
    for header in headers:
        table.add_column(header, justify="right")
    
    table.add_row("Parsing", *parse_times)
    table.add_row("Saving", *save_times)

    rich_print(table)

    print(f"Statistics time: {stats_t:.2f} s")
    print(f"Total benchmark time: {(stop - start):.2f} s")


if __name__ == "__main__":
    benchmark()