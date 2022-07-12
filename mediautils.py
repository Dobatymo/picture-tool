import logging
from typing import DefaultDict, Set, Tuple

from filemeta.mediainfo import MediaInfoFields
from genutility.exceptions import ParseError

logger = logging.getLogger()


def scandir_error_log(entry, exception):
    logger.error("%s in %s", exception.__class__.__qualname__, entry.path)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from collections import defaultdict
    from os import fspath

    # from pprint import pprint
    from genutility.args import is_dir
    from genutility.filesystem import fileextensions, scandir_ext
    from genutility.iter import progress

    parser = ArgumentParser()
    parser.add_argument("path", nargs="+", type=is_dir)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    try:
        import colorlog
    except ImportError:
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    else:
        formatter = colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.verbose:
        handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    mif = MediaInfoFields()

    extensions = {"." + ext for ext in fileextensions.video + fileextensions.audio + fileextensions.images}
    unhandled_keys = defaultdict(set)  # type: DefaultDict[str, Set[Tuple[str, str]]]

    for path in args.path:
        for path in progress(scandir_ext(path, extensions, errorfunc=scandir_error_log)):
            try:
                values = dict(mif.mediainfo(fspath(path), unhandled_keys))
            except ParseError as e:
                logger.error("%s failed to parse: %s", path, e)
                continue

            # pprint(path)
            # pprint(values)
            # print("---")

    mif.persist()
