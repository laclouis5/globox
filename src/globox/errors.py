from pathlib import Path


class UnknownImageFormat(Exception):
    pass


class ParsingError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


class FileParsingError(ParsingError):
    def __init__(self, file: Path, reason: str) -> None:
        self.file = file
        self.reason = reason

    def __str__(self) -> str:
        return f"Error while reading file '{self.file}': {self.reason}"
