import logging
from os import fspath
from pathlib import Path

import cv2
import dlib
import face_recognition
from genutility.concurrency import parallel_map
from genutility.pickle import read_pickle, write_pickle


class PictureWindow(object):

	def __init__(self, name="picture"):
		self.name = name
		cv2.startWindowThread()
		cv2.namedWindow("picture", cv2.WINDOW_NORMAL)

	def __enter__(self):
		return self

	def __exit__(self, *args):
		cv2.destroyAllWindows()

	def show(self, im, size=None, from_="rgb"):
		# type: (np.ndarray, Optional[Tuple[int, int]], str) -> None

		if size:
			im = cv2.resize(im, size)

		from_ = from_.lower()

		if from_ == "rgb":
			pass
		elif from_ in ("bgr", "pillow"):
			im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		else:
			raise ValueError()

		cv2.imshow(self.name, im)
		cv2.waitKey(1)

	def showfile(self, path):
		im = cv2.imread("earth.jpg")
		self.show(im)

def crop(img, loc, context=(0, 0, 0, 0)):
	top, right, bottom, left = loc
	t, r, b, l = context

	height, width, channels = img.shape

	top = max(0, top-t)
	right = min(width, right+r)
	bottom = min(height, bottom+b)
	left = max(0, left-l)

	return img[top:bottom, left:right]

class FaceStorage(object):

	def __init__(self):
		self._encodings = []
		self._names = []

		self._versions = self._get_versions()

	@staticmethod
	def _get_versions():
		return {
			"dlib": dlib.__version__,
			"face_recognition": face_recognition.__version__,
		}

	def add(self, path, encoding, name):
		self._encodings.append(encoding)
		self._names.append(name)

	@property
	def encodings(self):
		return self._encodings

	@property
	def names(self):
		return self._names

	def __getstate__(self):
		state = self.__dict__.copy()
		return state

	def __setstate__(self, state):

		if state["_versions"] != self._get_versions():
			logging.warning("dlib and face_recognition versions of pickled object and current environment don't match")

		self.__dict__.update(state)

def get_features(entry):

	img = face_recognition.load_image_file(fspath(entry))

	locations = face_recognition.face_locations(img)
	encodings = face_recognition.face_encodings(img, locations, num_jitters=1)
	# print(f"Found {len(locations)} face(s) in {fspath(entry)}")

	assert len(locations) == len(encodings)

	images = [crop(img, loc, (10, 10, 10, 10)) for loc in locations]

	return entry, images, encodings

def get_faces(paths):
	# type: (Iterable[DirEntry]) -> Iterator[Tuple[DirEntry, np.ndarray, np.ndarray]]

	try:
		for entry, images, encodings in parallel_map(get_features, paths, bufsize=100):
			for img, enc in zip(images, encodings):
				yield entry, img, enc
	except KeyboardInterrupt:
		print("Face analysis interrupted")
		raise

def label(paths, db, strict_bound, suggest_bound):
	# type: (Iterable[DirEntry], FaceStorage, float, float) -> None

	with PictureWindow() as win:

		for entry, img, enc in get_faces(paths):

			dists = face_recognition.face_distance(db.encodings, enc)
			suggest = [(i, d) for i, d in enumerate(dists) if d <= suggest_bound]
			sure = [i for i, d in suggest if d <= strict_bound]

			foundnames = {db.names[i] for i in sure}
			if len(foundnames) == 1:
				print(f"{entry.name} - Found: {foundnames.pop()}")
				continue
			elif len(foundnames) > 1:
				pass
			else:
				foundnames = {db.names[i] for i, d in suggest}

			if foundnames:
				foundnames = ", ".join(foundnames)
				print(f"{entry.name} - Matches: {foundnames}")

			win.show(img, from_="BGR")
			name = input(f"{entry.name} - Name: ")

			db.add(entry, enc, name)

if __name__ == "__main__":

	from argparse import ArgumentParser

	from genutility.args import is_dir

	DEFAULT_STRICT_BOUND = 0.1
	DEFAULT_SUGGEST_BOUND = 0.5
	DEFAULT_FACES_DB = "faces.p"

	parser = ArgumentParser()
	parser.add_argument("path", type=is_dir)
	parser.add_argument("--strict", type=float, default=DEFAULT_STRICT_BOUND)
	parser.add_argument("--suggest", type=float, default=DEFAULT_SUGGEST_BOUND)
	parser.add_argument("--faces-db", type=Path, default=DEFAULT_FACES_DB)
	parser.add_argument("-r", "--recursive", action="store_true")
	parser.add_argument("--verbose", action="store_true")
	args = parser.parse_args()

	if args.verbose:
		logging.basicConfig(level=logging.DEBUG)
	else:
		logging.basicConfig(level=logging.INFO)

	if args.recursive:
		it = Path(args.path).rglob("*.jpg")
	else:
		it = Path(args.path).glob("*.jpg")

	try:
		db = read_pickle(args.faces_db)
	except FileNotFoundError:
		db = FaceStorage()

	try:
		label(it, db, args.strict, args.suggest)
	except KeyboardInterrupt:
		print("Interrupted labeling")

	write_pickle(db, args.faces_db, safe=True)
