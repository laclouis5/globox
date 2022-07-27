import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    
    # General options
    parser.add_argument("--verbose", "-v", action="store_true")

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
    group.add_argument("--source", "-i", type=str, choices=("yolo", "voc", "coco"))
    group.add_argument("--img_folder", '-d', type=Path, default=None)
    group.add_argument("--mapping", "-m", type=Path, default=None)
    group.add_argument("--format", "-f", type=str, choices=("ltrb", "ltwh", "xywh"), default="ltrb", dest="format_in")
    group.add_argument("--norm", "-n", type=str, choices=("abs", "rel"), default="abs", dest="norm_in")
    group.add_argument("--ext", "-e", type=str, default=".txt")
    group.add_argument("--img_ext", "-g", type=str, default=".jpg")
    group.add_argument("--sep", "-s", type=str, default=" ")


def add_parse_gts_args(parser: argparse.ArgumentParser):
    parser.add_argument("groundtruths", type=Path)

    group = parser.add_argument_group("Ground-truths parse options")
    group.add_argument("--target", "-t", type=str, choices=("yolo", "voc", "coco"))
    group.add_argument("--img_folder_gts", '-D', type=Path, default=None)
    group.add_argument("--mapping_gts", "-M", type=Path, default=None)
    group.add_argument("--format_gts", "-F", type=str, choices=("ltrb", "ltwh", "xywh"), default="ltrb", dest="format_in_gts")
    group.add_argument("--norm_gts", "-N", type=str, choices=("abs", "rel"), default="abs", dest="norm_in_gts")
    group.add_argument("--ext_gts", "-E", type=str, default=".txt")
    group.add_argument("--img_ext_gts", "-G", type=str, default=".jpg")
    group.add_argument("--sep_gts", "-S", type=str, default=" ")


def add_save_args(parser: argparse.ArgumentParser):
    parser.add_argument("output", type=Path)

    group = parser.add_argument_group("Save options")
    group.add_argument("--target", "-o", type=str, choices=("yolo", "voc", "coco"))
    group.add_argument("--format_out", "-F", type=str, choices=("ltrb", "ltwh", "xywh"), default="ltrb", dest="format_out")
    group.add_argument("--norm_out", "-N", type=str, choices=("abs", "rel"), default="abs", dest="norm_out")
    group.add_argument("--sep_out", "-S", type=str, default=" ")
    group.add_argument("--ext_out", "-E", type=str, default=" ")
    group.add_argument("--mapping_out", "-M", type=Path, default=None)


def add_convert_args(parser: argparse.ArgumentParser): 
    add_parse_args(parser)
    add_save_args(parser)


def add_stats_args(parser: argparse.ArgumentParser):
    add_parse_args(parser)


def add_eval_args(parser: argparse.ArgumentParser):
    add_parse_args(parser, label="predictions")
    add_parse_gts_args(parser)
    # check sys.stdout.isatty() to redirect csv summary to file


def main():
    args = parse_args()
    print(args)


if __name__ == "__main__":
    main()