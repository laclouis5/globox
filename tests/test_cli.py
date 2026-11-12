import sys
from pathlib import Path

import pytest

from globox import Annotation, AnnotationSet, BoundingBox
from globox.cli import main


def test_convert_to_imagenet_writes_xml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    box = BoundingBox(label="cat", xmin=10.0, ymin=20.0, xmax=110.0, ymax=80.0)
    annotations = AnnotationSet(
        [Annotation(image_id="image.jpg", image_size=(200, 100), boxes=[box])]
    )
    input_path = tmp_path / "coco.json"
    annotations.save_coco(input_path, auto_ids=True)
    output_path = tmp_path / "imagenet"
    argv = ["globox", "--quiet", "convert", "-f", "coco", "-F", "imagenet"]
    monkeypatch.setattr(sys, "argv", [*argv, str(input_path), str(output_path)])

    main()

    parsed = AnnotationSet.from_imagenet(output_path)
    assert [box.label for box in parsed.all_boxes] == ["cat"]
