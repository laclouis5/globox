import os
import tempfile as tmp
from contextlib import contextmanager
from typing import Optional

from .file_utils import PathLike


@contextmanager
def _tempfile(suffix: str = "~", dir: Optional[PathLike] = None):
    tmp_file = tmp.NamedTemporaryFile(delete=False, suffix=suffix, dir=dir)
    tmp_name = tmp_file.name
    tmp_file.file.close()

    try:
        yield tmp_name
    finally:
        try:
            os.remove(tmp_name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise


@contextmanager
def open_atomic(file_path: PathLike, *args, **kwargs):
    fsync = kwargs.pop("fsync", False)

    with _tempfile(dir=os.path.dirname(os.path.abspath(file_path))) as tmp_path:
        with open(tmp_path, *args, **kwargs) as file:
            try:
                yield file
            finally:
                if fsync:
                    file.flush()
                    os.fsync(file.fileno())

        os.replace(tmp_path, file_path)
