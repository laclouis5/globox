from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]


def glob(
    folder: PathLike, extensions: Union[str, Iterable[str]], recursive: bool = False
) -> Iterable[Path]:
    """Glob files by providing extensions to match."""
    if isinstance(extensions, str):
        extensions = {extensions}
    else:
        extensions = set(extensions)

    assert all(
        e.startswith(".") for e in extensions
    ), f"Parameter `extensions' ({extensions}) should all start with a dot."

    path = Path(folder).expanduser().resolve()

    files = path.glob("**/*") if recursive else path.glob("*")

    return (f for f in files if f.suffix in extensions and not f.name.startswith("."))
