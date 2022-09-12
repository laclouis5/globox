from pathlib import Path
from setuptools import setup


NAME = "globox"
AUTHOR = "Louis Lac"
EMAIL = "lac.louis5@gmail.com"
URL = "https://github.com/laclouis5/globox"
DESCRIPTION = "Globox is a package and command line interface to read and convert object detection databases (COCO, YOLO, PascalVOC, LabelMe, CVAT, OpenImage, ...) and evaluate them with COCO and PascalVOC."

LICENSE = "MIT"
PYTHON = ">=3.8.2"

REQUIREMENTS = ["rich", "tqdm", "numpy"]

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

with (Path(NAME) / "__version__.py").open() as f:
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
    packages=[NAME],
    python_requires=PYTHON,
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [f"{NAME}={NAME}.cli:main"],
    }
)