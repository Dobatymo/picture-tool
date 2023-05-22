import logging
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict

import piexif
import pyexiv2
from genutility.args import is_dir
from genutility.filesystem import scandir_ext

from picturetool.utils import Max, extensions_jpeg, get_exif_dates, with_stem

modelmap = {
    b"iPhone SE (2nd generation)": "iPhone SE 2",
}


def get_items(path: Path) -> Dict[str, Any]:
    exif_dict = piexif.load(os.fspath(path))

    maker = exif_dict["0th"][271].decode("ascii")
    model = exif_dict["0th"][272]
    model = modelmap.get(model, model.decode("ascii"))
    dt_orig = get_exif_dates(exif_dict)["original"]

    return {
        "filename": path.name,
        "folder": path.parent,
        "maker": maker.replace(" ", "-"),
        "model": model.replace(" ", "-"),
        "date": dt_orig.date().isoformat(),
        "time": dt_orig.time().isoformat().replace(":", "-"),
        "datetime": dt_orig,
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
    parser.add_argument("--extensions", nargs="+", default=extensions_jpeg)
    args = parser.parse_args()

    files = {}

    # gather info
    for entry in scandir_ext(args.path, args.extensions):
        path = Path(entry)
        with pyexiv2.Image(os.fspath(path)) as img:
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

            def keyfunc(x):
                return x[1].get(args.sort_by, DEFAULT_MISSING)

        else:

            def keyfunc(x):
                return x[1][args.sort_by]

        try:
            _files = sorted(((path, items) for path, items in files.items()), key=keyfunc)
        except KeyError:
            logging.critical(f"Sort key `{args.sort_by}` not available for file <{path.name}>")
            sys.exit(1)

        for i, (path, items) in enumerate(_files):
            items["count"] = i
            try:
                newpath = with_stem(path, args.tpl.format_map(items))
            except KeyError:
                newpath = path.with_suffix(".stripped" + path.suffix)

            print(path.name, "->", newpath.name)


if __name__ == "__main__":
    main()
