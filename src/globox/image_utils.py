# Source: https://github.com/scardine/image_size

from os import path
from pathlib import Path
from struct import error as struct_error
from struct import unpack

from .errors import UnknownImageFormat
from .file_utils import PathLike

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".jpe",
    ".tif",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".JPE",
    ".TIF",
    ".TIFF",
]


def get_image_size(file_path: PathLike) -> "tuple[int, int]":
    """
    Compute the size of an image without loading into memory, which could result in faster speed.

    Parameters:

    * `file_path`: path to an image file.

    Returns:

    * The image size (width, height).
    """
    file_path = Path(file_path).expanduser().resolve()

    size = path.getsize(file_path)

    # be explicit with open arguments - we need binary mode
    with file_path.open("rb") as input:
        try:
            return _get_image_metadata_from_bytesio(input, size)
        except Exception as e:
            raise UnknownImageFormat(str(e))


def _get_image_metadata_from_bytesio(input, size: int) -> "tuple[int, int]":
    """
    Args:
        input (io.IOBase): io object support read & seek
        size (int): size of buffer in byte
        file_path (str): path to an image file
    Returns:
        Image: (path, type, file_size, width, height)
    """
    height = -1
    width = -1
    data = input.read(26)
    msg = " raised while trying to decode as JPEG."

    if (size >= 10) and data[:6] in (b"GIF87a", b"GIF89a"):
        # GIFs
        w, h = unpack("<HH", data[6:10])
        width = int(w)
        height = int(h)
    elif (
        (size >= 24)
        and data.startswith(b"\211PNG\r\n\032\n")
        and (data[12:16] == b"IHDR")
    ):
        # PNGs
        w, h = unpack(">LL", data[16:24])
        width = int(w)
        height = int(h)
    elif (size >= 16) and data.startswith(b"\211PNG\r\n\032\n"):
        # older PNGs
        w, h = unpack(">LL", data[8:16])
        width = int(w)
        height = int(h)
    elif (size >= 2) and data.startswith(b"\377\330"):
        # JPEG
        input.seek(0)
        input.read(2)
        b = input.read(1)
        try:
            while b and ord(b) != 0xDA:
                while ord(b) != 0xFF:
                    b = input.read(1)
                while ord(b) == 0xFF:
                    b = input.read(1)
                if ord(b) >= 0xC0 and ord(b) <= 0xC3:
                    input.read(3)
                    h, w = unpack(">HH", input.read(4))
                    width = int(w)
                    height = int(h)
                    break
                else:
                    input.read(int(unpack(">H", input.read(2))[0]) - 2)
                b = input.read(1)
        except struct_error:
            raise UnknownImageFormat("StructError" + msg)
        except ValueError:
            raise UnknownImageFormat("ValueError" + msg)
        except Exception as e:
            raise UnknownImageFormat(e.__class__.__name__ + msg)
    elif (size >= 26) and data.startswith(b"BM"):
        # BMP
        headersize = unpack("<I", data[14:18])[0]
        if headersize == 12:
            w, h = unpack("<HH", data[18:22])
            width = int(w)
            height = int(h)
        elif headersize >= 40:
            w, h = unpack("<ii", data[18:26])
            width = int(w)
            # as h is negative when stored upside down
            height = abs(int(h))
        else:
            raise UnknownImageFormat("Unkown DIB header size:" + str(headersize))
    elif (size >= 8) and data[:4] in (b"II\052\000", b"MM\000\052"):
        # Standard TIFF, big- or little-endian
        # BigTIFF and other different but TIFF-like formats are not
        # supported currently
        byteOrder = data[:2]
        boChar = ">" if byteOrder == "MM" else "<"
        # maps TIFF type id to size (in bytes)
        # and python format char for struct
        tiffTypes = {
            1: (1, boChar + "B"),  # BYTE
            2: (1, boChar + "c"),  # ASCII
            3: (2, boChar + "H"),  # SHORT
            4: (4, boChar + "L"),  # LONG
            5: (8, boChar + "LL"),  # RATIONAL
            6: (1, boChar + "b"),  # SBYTE
            7: (1, boChar + "c"),  # UNDEFINED
            8: (2, boChar + "h"),  # SSHORT
            9: (4, boChar + "l"),  # SLONG
            10: (8, boChar + "ll"),  # SRATIONAL
            11: (4, boChar + "f"),  # FLOAT
            12: (8, boChar + "d"),  # DOUBLE
        }
        ifdOffset = unpack(boChar + "L", data[4:8])[0]
        try:
            countSize = 2
            input.seek(ifdOffset)
            ec = input.read(countSize)
            ifdEntryCount = unpack(boChar + "H", ec)[0]
            # 2 bytes: TagId + 2 bytes: type + 4 bytes: count of values + 4
            # bytes: value offset
            ifdEntrySize = 12
            for i in range(ifdEntryCount):
                entryOffset = ifdOffset + countSize + i * ifdEntrySize
                input.seek(entryOffset)
                tag = input.read(2)
                tag = unpack(boChar + "H", tag)[0]
                if tag == 256 or tag == 257:
                    # if type indicates that value fits into 4 bytes, value
                    # offset is not an offset but value itself
                    type = input.read(2)
                    type = unpack(boChar + "H", type)[0]
                    if type not in tiffTypes:
                        raise UnknownImageFormat("Unkown TIFF field type:" + str(type))
                    typeSize = tiffTypes[type][0]
                    typeChar = tiffTypes[type][1]
                    input.seek(entryOffset + 8)
                    value = input.read(typeSize)
                    value = int(unpack(typeChar, value)[0])
                    if tag == 256:
                        width = value
                    else:
                        height = value
                if width > -1 and height > -1:
                    break
        except Exception as e:
            raise UnknownImageFormat(str(e))
    elif size >= 2:
        # see http://en.wikipedia.org/wiki/ICO_(file_format)
        input.seek(0)
        reserved = input.read(2)
        if 0 != unpack("<H", reserved)[0]:
            raise UnknownImageFormat("Sorry, don't know how to get size for this file")
        format = input.read(2)
        assert 1 == unpack("<H", format)[0]
        num = input.read(2)
        num = unpack("<H", num)[0]
        if num > 1:
            import warnings

            warnings.warn("ICO File contains more than one image")
        # http://msdn.microsoft.com/en-us/library/ms997538.aspx
        w = input.read(1)
        h = input.read(1)
        width = ord(w)
        height = ord(h)
    else:
        raise UnknownImageFormat("Sorry, don't know how to get size for this file")

    return width, height
