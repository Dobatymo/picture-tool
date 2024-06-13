import logging
from argparse import ArgumentParser
from collections import defaultdict
from os import fspath
from typing import DefaultDict, Set, Tuple

from filemeta.mediainfo import MediaInfoFields

# from pprint import pprint
from genutility.args import is_dir
from genutility.exceptions import ParseError
from genutility.filesystem import fileextensions, scandir_ext
from genutility.rich import Progress
from rich.logging import RichHandler
from rich.progress import Progress as RichProgress

logger = logging.getLogger()


def scandir_error_log(entry, exception):
    logger.error("%s in %s", exception.__class__.__qualname__, entry.path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", nargs="+", type=is_dir)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    handler = RichHandler(log_time_format="%Y-%m-%d %H-%M-%S%Z")
    FORMAT = "%(message)s"

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, handlers=[handler])
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT, handlers=[handler])

    mif = MediaInfoFields()

    extensions = {"." + ext for ext in fileextensions.video + fileextensions.audio + fileextensions.images}
    unhandled_keys: DefaultDict[str, Set[Tuple[str, str]]] = defaultdict(set)

    with RichProgress() as progress:
        p = Progress(progress)
        for basepath in args.path:
            for path in p.track(scandir_ext(basepath, extensions, errorfunc=scandir_error_log)):
                try:
                    values = dict(mif.mediainfo(fspath(path), unhandled_keys))
                except ParseError as e:
                    logger.error("%s failed to parse: %s", path, e)
                    continue

                # pprint(path)
                # pprint(values)
                # print("---")

    mif.persist()
