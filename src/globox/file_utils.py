from pathlib import Path
from typing import Union, Iterable


def glob(
    folder: Path, 
    extensions: Union[str, Iterable[str]], 
    recursive: bool = False
) -> Iterable[Path]:
    """Glob files by providing extensions to match."""
    if isinstance(extensions, str):
        extensions = {extensions}
    else:
        extensions = set(extensions)

    assert all(e.startswith(".") for e in extensions), \
        "Parameter `extension` should start with a dot."

    files = folder.glob("**/*") if recursive else folder.glob("*")

    return (f for f in files \
        if f.suffix in extensions and not f.name.startswith("."))