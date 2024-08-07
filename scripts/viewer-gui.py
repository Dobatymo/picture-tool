import logging
import os.path
import sys
from logging.handlers import TimedRotatingFileHandler
from multiprocessing.connection import Client
from pathlib import Path
from typing import Optional

from platformdirs import user_config_dir, user_log_dir

APP_AUTHOR = "Dobatymo"
APP_NAME = "picture-viewer"
APP_PIPE_NAME = rf"\\.\pipe\{APP_AUTHOR}-{APP_NAME}"


def try_send(name: str, msg: Optional[dict]) -> None:
    if os.path.exists(name):
        with Client(name, "AF_PIPE") as conn:
            conn.send(msg)
        sys.exit(0)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from genutility.args import existing_path

    parser = ArgumentParser()
    parser.add_argument(
        "paths",
        metavar="PATH",
        type=existing_path,
        nargs="*",
        help="Image files or folders to open. If a single file is specified, the other images from that folder will be added as context. If multiple files or folders are specified only they or there contents will be loaded, no other context.",
    )
    parser.add_argument(
        "--mode",
        choices=("fit", "original"),
        default="fit",
        help="Resize files to fit the window, or show in original resolution",
    )
    parser.add_argument(
        "--resolve-city-names", action="store_true", help="Resolve exif GPS coordinates to city names (slow)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log additional information useful for debugging")
    args = parser.parse_args()

    log_fmt = "%(asctime)s %(levelname)s:%(name)s:%(funcName)s:%(thread)d:%(message)s"

    filename = Path(user_log_dir(APP_NAME, APP_AUTHOR)) / "viewer-gui.log"
    filename.parent.mkdir(parents=True, exist_ok=True)
    stream_handler = logging.StreamHandler()
    file_handler = TimedRotatingFileHandler(filename, "midnight", encoding="utf-8", delay=True, utc=False)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=log_fmt, handlers=[stream_handler, file_handler])
        logging.getLogger("PIL").setLevel(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format=log_fmt, handlers=[stream_handler, file_handler])

    args.paths = [p.resolve(strict=True) for p in args.paths]

    os.chdir(Path.home())

    try_send(APP_PIPE_NAME, vars(args))

    # `try_send` will exit the process if the app is already running.
    # These imports are afterwards, so they don't have to be loaded when the app is already running
    # and thus speed up the file open process.
    import json

    from PySide2 import QtCore, QtWidgets

    from picturetool.viewer_gui import PictureWindow, PyServer, WindowManager

    def excepthook(exc_type, value, traceback):
        logging.exception("Unhandled exception", exc_info=(exc_type, value, traceback))

    class MyWindowManager(WindowManager):
        @QtCore.Slot()
        def about(self) -> None:
            QtWidgets.QMessageBox.about(
                None,
                "About",
                f"Open windows: {len(self.windows)}",
            )

    sys.excepthook = excepthook

    app = QtWidgets.QApplication([])

    configfile = Path(user_config_dir(APP_NAME, APP_AUTHOR, roaming=True)) / "config.json"
    try:
        with open(configfile, encoding="utf-8") as fr:
            config = json.load(fr)
    except FileNotFoundError:
        config = {}
    except json.JSONDecodeError as e:
        QtWidgets.QMessageBox.warning(
            None,
            "Could read config file",
            f"Config file <{configfile}> failed to parse: {e}",
        )
        ret = app.exec_()
        sys.exit(ret)

    wm = MyWindowManager(APP_NAME, PictureWindow)
    s = PyServer(APP_PIPE_NAME)
    s.start()
    s.message_received.connect(wm.create_from_args)

    wm.make_tray(app)
    wm.set_create_args(maximized=True, resolve_city_names=args.resolve_city_names, mode=args.mode, config=config)

    window = wm.create()

    # only load images after Qt loop is up and running
    QtCore.QTimer.singleShot(0, lambda: window.load_pictures(args.paths))

    ret = app.exec_()
    s.stop()
    if not s.wait(1000):
        logging.error("Background thread failed to exit in time")

    logging.debug("App exit code: %d", ret)
    sys.exit(ret)
