import logging
import os
from pathlib import Path
from typing import Iterable, Iterator, Union

import pandas as pd
from filemeta.exif import InvalidImageDataError, exif_table
from genutility.cache import cache
from genutility.datetime import datetime_from_utc_timestamp_ns
from pandasql import sqldf

EntryLike = Union[Path, os.DirEntry]

logger = logging.getLogger(__name__)

ALL_COLUMNS = [
    "0th Artist",
    "0th Bits per sample",
    "0th Compression",
    "0th Copyright",
    "0th DateTime",
    "0th ExifTag",
    "0th GPSTag",
    "0th Host computer",
    "0th Image description",
    "0th Image length",
    "0th Image width",
    "0th Make",
    "0th Model",
    "0th Orientation",
    "0th Photometric interpretation",
    "0th Planar configuration",
    "0th Processing software",
    "0th Rating",
    "0th RatingPercent",
    "0th Resolution unit",
    "0th Samples per pixel",
    "0th Software",
    "0th Tile length",
    "0th Tile width",
    "0th YCbCr positioning",
    "1st Compression",
    "1st Image length",
    "1st Image width",
    "1st JPEGInterchangeFormat",
    "1st JPEGInterchangeFormatLength",
    "1st Orientation",
    "1st Resolution unit",
    "1st YCbCr positioning",
    "Exif Body serial number",
    "Exif Color space",
    "Exif Contrast",
    "Exif Custom rendered",
    "Exif DateTime digitized",
    "Exif DateTime original",
    "Exif Exposure mode",
    "Exif Exposure program",
    "Exif Flash",
    "Exif Focal length in 35mm film",
    "Exif Focal plane resolution unit",
    "Exif Gain control",
    "Exif ISO speed ratings",
    "Exif Image unique ID",
    "Exif Interoperability tag",
    "Exif Lens make",
    "Exif Lense model",
    "Exif Light source",
    "Exif Metering mode",
    "Exif Offset time",
    "Exif Offset time digitized",
    "Exif Offset time original",
    "Exif Pixel X dimension",
    "Exif Pixel Y dimension",
    "Exif Recommended exposure index",
    "Exif Saturation",
    "Exif Scene capture type",
    "Exif Sensing method",
    "Exif Sensitivity type",
    "Exif Sharpness",
    "Exif Sub seconds time",
    "Exif Sub seconds time digitized",
    "Exif Sub seconds time original",
    "Exif Subject area",
    "Exif Subject distance range",
    "Exif White balance",
    "File size",
    "GPS GPS DateStamp",
    "GPS GPS destination bearing reference",
    "GPS GPS geodetic datum",
    "GPS GPS image direction reference",
    "GPS GPS latitude reference",
    "GPS GPS longitude reference",
    "GPS GPS satellites",
    "GPS GPS speed reference",
    "File modification time",
]


def get_dataframe_by_iter(pathiter: Iterable[EntryLike]) -> pd.DataFrame:
    values = []
    for path in pathiter:
        filepath = os.fspath(path)
        filesize = path.stat().st_size
        mtime = path.stat().st_mtime_ns

        values.append([filepath, "filesize", "File size", filesize, filesize])
        values.append(
            [filepath, "mtime", "File modification time", mtime, datetime_from_utc_timestamp_ns(mtime, aslocal=True)]
        )

        try:
            for ifd, key, key_label, value, value_label in exif_table(filepath):
                values.append([filepath, ifd + "-" + key, ifd + " " + key_label, value, value_label])
        except InvalidImageDataError as e:
            logger.warning("%s is not an image file with valid exif data: %s", path, e)

    df = pd.DataFrame(values, columns=("path", "key", "key_label", "value", "value_label"))
    df = df.pivot(index="path", columns="key_label", values="value_label")
    return df


def get_dataframe_by_path(path: Path) -> pd.DataFrame:
    extensions = {".jpg", ".jpeg", ".tif", ".tiff", ".webp"}  # ".wav", ".png" not supported by piexif

    def it(path: Path) -> Iterator[Path]:
        for p in path.rglob("*"):
            if p.suffix.lower() in extensions:
                yield p

    return get_dataframe_by_iter(it(path))


if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from datetime import timedelta

    from genutility.args import is_dir

    epilog = r"""Examples

- `%(prog)s "C:\Users\Public\Pictures" --groupby "0th Model"`

Show a list of all different camera models found with a count of how often they are used.

- `%(prog)s "C:\Users\Public\Pictures" --filter "0th Model" "iPhone XR" --select path`

Find all pictures taken with an iPhone XR and show the file paths.

- ``%(prog)s --sql "SELECT path, `0th Make`, `0th Model` FROM df WHERE `0th Model` LIKE '%%canon eos%%'" D:\``

Find all pictures taken by a Canon EOS camera using SQL query.
"""

    parser = ArgumentParser(
        description="Calculate exif statistics from picture files",
        epilog=epilog,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", metavar="PATH", nargs="+", type=is_dir, help="Input directories to scan")
    parser.add_argument("--rebuild-cache", action="store_true", help="Forces a cache rebuild")
    parser.add_argument("--verbose", action="store_true", help="Debug output")
    parser.add_argument("--cache-path", metavar="PATH", type=Path, default=Path("cache"), help="Path to cache")
    parser.add_argument("--select", metavar="FEATURE", nargs="+", type=str, help="Features to include in output")
    parser.add_argument("--sort-values", action="store_true", help="Sort output by value instead of key")
    parser.add_argument("--out", metavar="PATH", type=Path, help="If given write csv output to file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--groupby",
        metavar="FEATURE",
        choices=ALL_COLUMNS,
        help=f"Collect statistics for feature. Allowed features are: {', '.join(ALL_COLUMNS)}",
    )
    group.add_argument(
        "--filter",
        metavar=("FEATURE", "VALUE"),
        nargs=2,
        help='Keep only files where FEATURE equals VALUE, ie. "0th Model" "iPhone XR"',
    )
    group.add_argument("--sql", metavar="QUERY", help="Query by SQL. Use table `df`.")
    args = parser.parse_args()

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.rebuild_cache:
        df = pd.concat(map(cache(args.cache_path, duration=timedelta(0))(get_dataframe_by_path), args.paths))
    else:
        df = pd.concat(map(cache(args.cache_path)(get_dataframe_by_path), args.paths))

    df = df[~df.index.duplicated(keep="first")]

    if args.groupby:
        if args.select:
            parser.error("--select can only be used with --filter")

        if args.groupby not in df.columns:
            parser.error(f"--groupby value {args.groupby!r} not in data")

        result = df.fillna("N/A").groupby(args.groupby).size()
        if args.sort_values:
            result.sort_values(inplace=True)

    elif args.filter:
        if args.sort_values:
            parser.error("--sort-values can only be used with --groupby")

        key, value = args.filter
        if key not in df.columns:
            parser.error(f"--filter KEY={key!r} not in data")

        result = df[df[key] == value].reset_index()
        if args.select:
            result = result[args.select]
        result.dropna(axis="columns", how="all", inplace=True)

    elif args.sql:
        """alternative:
        import duckdb
        result = duckdb.sql(args.sql).df()
        """
        result = sqldf(args.sql, {"df": df})

    if args.out:
        result.to_csv(args.out, index=False)
    else:
        result.columns.name = None
        print(result)
