import argparse
from pathlib import Path
from ObjectDetectionEval import *


PARSE_CHOICES = {"coco", "yolo", "labelme", "pascalvoc", "openimage", "txt", "cvat"}


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--threads", "-j", type=int, default=0)

    subparsers = parser.add_subparsers(dest="mode")
    convert_parser = subparsers.add_parser("convert")
    stats_parser = subparsers.add_parser("summary")
    eval_parser = subparsers.add_parser("evaluate")

    add_convert_args(convert_parser)
    add_stats_args(stats_parser)
    add_eval_args(eval_parser)

    return parser.parse_args()


def add_parse_args(parser: argparse.ArgumentParser, label: str = "input"):
    parser.add_argument("input", type=Path, metavar=label)
    
    group = parser.add_argument_group("Parse options" if label == "input" else "Predictions parse options")
    group.add_argument("--format", "-f", type=str, choices=PARSE_CHOICES, dest="format_in")
    group.add_argument("--img_folder", '-d', type=Path, default=None)
    group.add_argument("--mapping", "-m", type=Path, default=None, dest="mapping_in")
    group.add_argument("--bb_fmt", "-b", type=str, choices=("ltrb", "ltwh", "xywh"), default="ltrb", dest="bb_fmt_in")
    group.add_argument("--norm", "-n", type=str, choices=("abs", "rel"), default="abs", dest="norm_in")
    group.add_argument("--ext", "-e", type=str, default=".txt", dest="ext_in")
    group.add_argument("--img_ext", "-g", type=str, default=".jpg", dest="img_ext_in")
    group.add_argument("--sep", "-p", type=str, default=" ", dest="sep_in")


def add_parse_gts_args(parser: argparse.ArgumentParser):
    parser.add_argument("groundtruths", type=Path)

    group = parser.add_argument_group("Ground-truths parse options")
    group.add_argument("--target_fmt", "-F", type=str, choices=PARSE_CHOICES, dest="format_gts")
    group.add_argument("--img_folder_gts", '-D', type=Path, default=None)
    group.add_argument("--mapping_gts", "-M", type=Path, default=None)
    group.add_argument("--bb_fmt_gts", "-B", type=str, choices=("ltrb", "ltwh", "xywh"), default="ltrb", dest="bb_fmt_gts")
    group.add_argument("--norm_gts", "-N", type=str, choices=("abs", "rel"), default="abs", dest="norm_in_gts")
    group.add_argument("--ext_gts", "-E", type=str, default=".txt")
    group.add_argument("--img_ext_gts", "-G", type=str, default=".jpg")
    group.add_argument("--sep_gts", "-P", type=str, default=" ")


def add_save_args(parser: argparse.ArgumentParser):
    parser.add_argument("output", type=Path)

    group = parser.add_argument_group("Save options")
    group.add_argument("--save_fmt", "-F", type=str, choices=PARSE_CHOICES, dest="format_out")
    group.add_argument("--mapping_out", "-M", type=Path, default=None)
    group.add_argument("--bb_fmt_out", "-B", type=str, choices=("ltrb", "ltwh", "xywh"), default="ltrb")
    group.add_argument("--norm_out", "-N", type=str, choices=("abs", "rel"), default="abs")
    group.add_argument("--sep_out", "-P", type=str, default=" ")
    group.add_argument("--ext_out", "-E", type=str, default=" ")


def add_convert_args(parser: argparse.ArgumentParser): 
    add_parse_args(parser)
    add_save_args(parser)


def add_stats_args(parser: argparse.ArgumentParser):
    add_parse_args(parser)


def add_eval_args(parser: argparse.ArgumentParser):
    add_parse_args(parser, label="predictions")
    add_parse_gts_args(parser)

    # parser.add_argument("--ap", action="append", dest="metrics")
    # parser.add_argument("--ap50", action="append", dest="metrics")
    # parser.add_argument("--iou", type=int, default=None)  # mutually_exclusive_group()
    # etc...

def parse_annotations(args: argparse.Namespace) -> AnnotationSet:
    input: Path = args.input.expanduser().resolve()
    format_in: str = args.format_in
    image_dir: Path = args.img_folder

    if format_in == "coco":
        return AnnotationSet.from_coco(input)
    elif format_in == "pascalvoc":
        return AnnotationSet.from_xml(input)
    elif format_in == "openimage":
        return AnnotationSet.from_openimage(input, image_dir)
    elif format_in == "labelme":
        return AnnotationSet.from_labelme(input)
    elif format_in == "cvat":
        return AnnotationSet.from_cvat(input)
    else:
        img_ext: str = args.img_ext_in

        if format_in == "yolo":
            return AnnotationSet.from_yolo(input, image_dir, img_ext)
        elif format_in == "txt":
            format_in = BoxFormat.from_string(args.bb_fmt_in)
            relative: bool = args.norm_in == "rel"
            extension: str = args.ext_in
            sep: str = args.sep_in

            return AnnotationSet.from_txt(input, image_dir, format_in, relative, extension, img_ext, sep)
        else:
            raise ValueError(f"Input format '{format_in}' unknown")


def parse_gts_annotations(args: argparse.Namespace) -> AnnotationSet:
    input: Path = args.groundtruths.expanduser().resolve()
    format_gts: str = args.format_gts
    image_dir: Path = args.img_folder_gts

    if format_gts == "coco":
        return AnnotationSet.from_coco(input)
    elif format_gts == "pascalvoc":
        return AnnotationSet.from_xml(input)
    elif format_gts == "openimage":
        return AnnotationSet.from_openimage(input, image_dir)
    elif format_gts == "labelme":
        return AnnotationSet.from_labelme(input)
    elif format_gts == "cvat":
        return AnnotationSet.from_cvat(input)
    else:
        img_ext: str = args.img_ext_gts

        if format_gts == "yolo":
            return AnnotationSet.from_yolo(input, image_dir, img_ext)
        elif format_gts == "txt":
            bb_fmt = BoxFormat.from_string(args.bb_fmt_gts)
            relative: bool = args.norm_gts == "rel"
            extension: str = args.ext_gts
            sep: str = args.sep_gts

            return AnnotationSet.from_txt(input, image_dir, bb_fmt, relative, extension, img_ext, sep)
        else:
            raise ValueError(f"Groundtruth format '{format_gts}' unknown")


def save_annotations(args: argparse.Namespace, annotations: AnnotationSet):
    output: Path = args.output.expanduser().resolve()
    format_out: str = args.format_out

    if format_out == "coco":
        annotations.save_coco(output)
    elif format_out == "pascalvoc":
        annotations.save_xml(output)
    elif format_out == "openimage":
        annotations.save_openimage(output)
    elif format_out == "labelme":
        annotations.save_labelme(output)
    elif format_out == "cvat":
        annotations.save_cvat(output)
    else:
        if format_out == "yolo":
            annotations.save_yolo(output)  # TODO: Add output mapping.
        elif format_out == "txt":
            bb_fmt = BoxFormat.from_string(args.bb_fmt_out)
            relative: bool = args.norm_out == "rel"
            extension: str = args.ext_out
            sep: str = args.sep_out

            # TODO: Add output mapping.
            annotations.save_txt(output, None, bb_fmt, relative, sep, extension)
        else:
            raise ValueError(f"Save format '{format_out}' unknown")


def main():
    args = parse_args()

    annotations = parse_annotations(args)    

    mode = args.mode
    if mode == "convert":
        save_annotations(args, annotations)
    elif mode == "summary":
        annotations.show_stats()
    elif mode == "evaluate":
        groundtruths = parse_gts_annotations(args)
        evaluator = COCOEvaluator(groundtruths, annotations)
        evaluator.show_summary()
        # TODO: Add support for saving csv to file
    else:
        raise ValueError(f"Sub-command '{mode}' not recognized.")


if __name__ == "__main__":
    main()