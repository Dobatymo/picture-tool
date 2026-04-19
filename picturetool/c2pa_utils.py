import json

from c2pa import C2paError, Reader


def c2pa_json(path: str) -> dict:
    reader = Reader(path)
    return json.loads(reader.json())


__all__ = ["C2paError", "c2pa_json"]
