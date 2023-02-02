from pathlib import Path
from setuptools import setup, find_packages


NAME = "globox"
AUTHOR = "Louis Lac"
EMAIL = "lac.louis5@gmail.com"
URL = "https://github.com/laclouis5/globox"
DESCRIPTION = "Globox is a package and command line interface to read and convert object detection databases (COCO, YOLO, PascalVOC, LabelMe, CVAT, OpenImage, ...) and evaluate them with COCO and PascalVOC."

LICENSE = "MIT"
PYTHON = ">=3.7"

REQUIREMENTS = ["rich", "tqdm", "numpy"]
EXTRA_REQ = ["tox", "pytest", "twine", "build", "pycocotools", "Pillow"]

with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

with Path("src", NAME, "__version__.py").open(encoding="utf-8") as f:
    about = {}
    exec(f.read(), about)
    VERSION = about["__version__"]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=PYTHON,
    install_requires=REQUIREMENTS,
    extras_require={"dev": EXTRA_REQ},
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "annotation",
        "metrics",
        "object detection",
        "bounding boxes",
        "yolo",
        "openimages",
        "cvat",
        "coco",
        "pascal voc",
        "average precision",
        "mean average precision",
    ],
    entry_points={
        "console_scripts": [f"{NAME}={NAME}.cli:main"],
    },
)
