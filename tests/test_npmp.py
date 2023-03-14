import numpy as np
from genutility.test import MyTestCase, parametrize

from picturetool.npmp import ChunkedParallel, SharedNdarray
from picturetool.utils import array_from_iter, l2squared_duplicates_chunk, np_sorted, npmp_to_pairs, unique_pairs


def sum_chunk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a + b, axis=-1)


class SharedNdarrayTest(MyTestCase):
    @parametrize(
        ((1,), np.uint8, 1, 1),
        ((1,), np.uint64, 1, 8),
        ((2, 3), np.uint8, 6, 6),
        ((2, 3), np.uint64, 6, 48),
    )
    def test_create(self, shape, dtype, truth_size, truth_nbytes):
        arr = SharedNdarray.create(shape, dtype)

        self.assertEqual(arr.size, truth_size)
        self.assertEqual(arr.nbytes, truth_nbytes)
        self.assertGreaterEqual(arr.shm.size, truth_nbytes)

    @parametrize(
        ((1,), np.uint8, 1, 1),
        ((1,), np.uint64, 1, 8),
        ((2, 3), np.uint8, 6, 6),
        ((2, 3), np.uint64, 6, 48),
    )
    def test_from_array(self, shape, dtype, truth_size, truth_nbytes):
        arr = SharedNdarray.from_array(np.empty(shape, dtype))

        self.assertEqual(arr.size, truth_size)
        self.assertEqual(arr.nbytes, truth_nbytes)
        self.assertGreaterEqual(arr.shm.size, truth_nbytes)

    def test_str(self):
        text = str(SharedNdarray.create((1,), "uint16"))
        regex = r"<SharedNdarray\ shm\.name=[a-z]+_[a-z0-9]{8}\ shape=\(1,\)\ dtype=uint16\ shm\.buf=0000>"
        self.assertRegex(text, regex)

        text = str(SharedNdarray.create((2, 2), float))
        regex = r"<SharedNdarray\ shm\.name=[a-z]+_[a-z0-9]{8}\ shape=\(2, 2\)\ dtype=float64\ shm\.buf=00000000000000000000...>"
        self.assertRegex(text, regex)


class NpmpTest(MyTestCase):
    @classmethod
    def setUpClass(cls):
        seed = 0
        rng = np.random.default_rng(seed)

        cls.inputs = {}
        cls.truths_2d = {}
        cls.truths_3d = {}

        dims_2d = [1, 10, 100, 271, 997]
        dims_3d = [1, 2, 3, 89]
        cls.arr = rng.uniform(0, 1, size=(max(dims_2d + dims_3d), 10)).astype(np.float32)

        for d in dims_2d:
            cls.truths_2d[d] = l2squared_duplicates_chunk(cls.arr[:d, :], cls.arr[:d, :], threshold=1.0)

        for d in dims_3d:
            pairs = unique_pairs(l2squared_duplicates_chunk(cls.arr[None, :d, :], cls.arr[:d, None, :], threshold=1.0))
            cls.truths_3d[d] = np_sorted(pairs)

    @parametrize(
        (1, 1),
        (10, 1),
        (10, 10),
        (100, 1),
        (100, 10),
        (100, 100),
        (271, 137),
        (997, 101),
    )
    def test_mp_2d(self, d, chunksize):
        sharr = SharedNdarray.from_array(self.arr[:d, :])

        it = ChunkedParallel(
            l2squared_duplicates_chunk,
            sharr,
            sharr,
            (chunksize,),
            ordered=True,
            backend="multiprocessing",
            pass_coords=True,
            threshold=1.0,
        )
        result = array_from_iter(it)
        np.testing.assert_array_equal(result, self.truths_2d[d])

    @parametrize(
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
        (3, 3),
        (89, 23),
    )
    def test_mp_3d(self, d, chunksize):
        sharr = SharedNdarray.from_array(self.arr[:d, :])
        a_arr = sharr.reshape((1, sharr.shape[0], sharr.shape[1]))
        b_arr = sharr.reshape((sharr.shape[0], 1, sharr.shape[1]))

        it = ChunkedParallel(
            l2squared_duplicates_chunk,
            a_arr,
            b_arr,
            (chunksize, chunksize),
            ordered=True,
            backend="multiprocessing",
            pass_coords=True,
            threshold=1.0,
        )
        result = np_sorted(npmp_to_pairs(it))
        np.testing.assert_array_equal(result, self.truths_3d[d])

    @parametrize(
        (1, 1),
        (10, 1),
        (10, 10),
        (100, 1),
        (100, 10),
        (100, 100),
        (271, 137),
        (997, 101),
    )
    def test_mt_2d(self, d, chunksize):
        it = ChunkedParallel(
            l2squared_duplicates_chunk,
            self.arr[:d, :],
            self.arr[:d, :],
            (chunksize,),
            ordered=True,
            backend="threading",
            pass_coords=True,
            threshold=1.0,
        )
        result = array_from_iter(it)
        np.testing.assert_array_equal(result, self.truths_2d[d])

    @parametrize(
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
        (3, 3),
        (89, 23),
    )
    def test_mt_3d(self, d, chunksize):
        it = ChunkedParallel(
            l2squared_duplicates_chunk,
            self.arr[None, :d, :],
            self.arr[:d, None, :],
            (chunksize, chunksize),
            ordered=True,
            backend="threading",
            pass_coords=True,
            threshold=1.0,
        )
        result = np_sorted(npmp_to_pairs(it))
        np.testing.assert_array_equal(result, self.truths_3d[d])


if __name__ == "__main__":
    import unittest

    unittest.main()
