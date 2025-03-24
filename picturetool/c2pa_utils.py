import json
import os.path
from typing import Optional

from c2pa.c2pa import c2pa


def c2pa_json(path: str, format: Optional[str] = None) -> dict:
    reader = c2pa.Reader()
    with open(path, "rb") as fr:
        if format is None:
            format = os.path.splitext(path)[1][1:]
        reader.from_stream(format, C2paStream(fr))

    return json.loads(reader.json())


class C2paStream(c2pa.Stream):
    def __init__(self, stream) -> None:
        self.stream = stream

    def read_stream(self, length: int) -> bytes:
        return self.stream.read(length)

    def seek_stream(self, pos: int, mode: c2pa.SeekMode) -> int:
        whence = 0
        if mode is c2pa.SeekMode.CURRENT:
            whence = 1
        elif mode is c2pa.SeekMode.END:
            whence = 2
        return self.stream.seek(pos, whence)
