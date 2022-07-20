import logging
import re
import sys
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from functools import total_ordering
from os import fspath
from pathlib import Path
from typing import Any, DefaultDict, Dict

import piexif
import pyexiv2
from genutility.args import is_dir

modelmap = {
    b"iPhone SE (2nd generation)": "iPhone SE 2",
}


@total_ordering
class MaxType:
    def __le__(self, other):
        return False

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return "Max"


Max = MaxType()


def to_datetime(dt: bytes, offset: bytes, subsec: bytes) -> datetime:

    s = (dt + offset).decode("ascii")
    _subsec = subsec.decode("ascii")

    m = re.match(r"(\d\d\d\d):(\d\d):(\d\d) (\d\d):(\d\d):(\d\d)(\+|\-)(\d\d):(\d\d)", s)
    assert m
    ye, mo, da, ho, mi, se, sign, hooff, mioff = m.groups()
    return datetime.fromisoformat(f"{ye}-{mo}-{da}T{ho}:{mi}:{se}.{_subsec}{sign}{hooff}:{mioff}")


def with_stem(path: Path, stem: str) -> Path:
    return path.with_name(stem + path.suffix)


def get_items(path: Path) -> Dict[str, Any]:

    exif_dict = piexif.load(fspath(path))

    DateTimeOriginal = exif_dict["Exif"][36867]
    OffsetTimeOriginal = exif_dict["Exif"][36881]
    SubsecTimeOriginal = exif_dict["Exif"][37521]

    maker = exif_dict["0th"][271].decode("ascii")
    model = exif_dict["0th"][272]
    model = modelmap.get(model, model.decode("ascii"))
    dt = to_datetime(DateTimeOriginal, OffsetTimeOriginal, SubsecTimeOriginal)

    return {
        "filename": path.name,
        "folder": path.parent,
        "maker": maker.replace(" ", "-"),
        "model": model.replace(" ", "-"),
        "date": dt.date().isoformat(),
        "time": dt.time().isoformat().replace(":", "-"),
        "datetime": dt,
    }


def main() -> None:
    DEFAULT_TPL = "{count}_{date}_{model}"
    DEFAULT_SORT_BY = "filename"
    DEFAULT_GROUP_BY = "folder"
    DEFAULT_MISSING = Max

    parser = ArgumentParser()
    parser.add_argument("--tpl", type=str, default=DEFAULT_TPL)
    parser.add_argument("--group-by", type=str, default=DEFAULT_GROUP_BY)
    parser.add_argument("--sort-by", type=str, default=DEFAULT_SORT_BY)
    parser.add_argument("--path", type=is_dir, required=True)
    parser.add_argument("--no-fail-missing", action="store_true")
    args = parser.parse_args()

    files = {}

    # gather info
    for path in args.path.rglob("*.jpg"):
        with pyexiv2.Image(fspath(path)) as img:
            has_iptc = bool(img.read_iptc())
            has_exif = bool(img.read_exif())

        assert not has_iptc

        items = {
            "filename": path.name,
            "folder": path.parent,
        }

        if has_exif:
            try:
                exif = get_items(path)
            except KeyError:
                print(path)
            else:
                items.update(exif)

        files[path] = items

    # group
    if args.group_by:
        grouped: DefaultDict[Any, Dict[Path, Dict[str, Any]]] = defaultdict(dict)
        for path, items in files.items():
            if args.no_fail_missing:
                key = items.get(args.group_by, DEFAULT_MISSING)
            else:
                try:
                    key = items[args.group_by]
                except KeyError:
                    logging.critical(f"Group key `{args.group_by}` not available for file <{path.name}>")
                    sys.exit(1)

            grouped[key][path] = items
    else:
        grouped = {DEFAULT_MISSING: files}

    # sort
    for groupname, files in grouped.items():
        print(groupname)

        if args.no_fail_missing:
            keyfunc = lambda x: x[1].get(args.sort_by, DEFAULT_MISSING)
        else:
            keyfunc = lambda x: x[1][args.sort_by]

        try:
            _files = sorted(((path, items) for path, items in files.items()), key=keyfunc)
        except KeyError:
            logging.critical(f"Sort key `{args.sort_by}` not available for file <{path.name}>")
            sys.exit(1)

        for i, (path, items) in enumerate(_files):
            items["count"] = i
            try:
                newpath = with_stem(path, args.tpl.format(**items))
            except KeyError:
                newpath = path.with_suffix(".stripped" + path.suffix)

            print(path.name, "->", newpath.name)


if __name__ == "__main__":

    main()
