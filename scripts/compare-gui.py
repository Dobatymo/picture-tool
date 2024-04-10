# pip install pandas pillow pillow-heif PySide2 humanize
import logging
import os
import sys

from PySide2 import QtWidgets

from picturetool.compare_gui import TableWindow

APP_NAME = "compare-gui"
PIC_CACHE_SIZE = 4


if __name__ == "__main__":
    from argparse import ArgumentParser

    from genutility.args import is_file

    parser = ArgumentParser()
    parser.add_argument("--in-path", type=is_file, help="Allowed file types are csv, parquet and json")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log additional information useful for debugging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = QtWidgets.QApplication([])

    widget = TableWindow(APP_NAME, PIC_CACHE_SIZE)
    widget.setWindowTitle(APP_NAME)
    widget.resize(800, 600)
    if args.in_path:
        widget.read_file(os.fspath(args.in_path))
    widget.show()

    sys.exit(app.exec_())
