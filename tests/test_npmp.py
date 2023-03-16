import pickle  # nosec

import numpy as np
from genutility.test import MyTestCase, parametrize

from picturetool.npmp import ChunkedParallel, SharedNdarray
from picturetool.utils import array_from_iter, l2squared_duplicates_chunk, np_sorted, npmp_to_pairs, unique_pairs


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
        self.assertGreaterEqual(arr.shm.buf.nbytes, truth_nbytes)
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
        regex = r"<SharedNdarray\ shm\.name=[a-z]+_[a-z0-9]{8}\ shape=\(1,\)\ dtype=uint16\ strides=None shm\.buf=0000>"
        self.assertRegex(text, regex)

        text = str(SharedNdarray.create((2, 2), float))
        regex = r"<SharedNdarray\ shm\.name=[a-z]+_[a-z0-9]{8}\ shape=\(2, 2\)\ dtype=float64\ strides=None shm\.buf=00000000000000000000...>"
        self.assertRegex(text, regex)

    @parametrize(
        (np.arange(16, dtype=np.int32).reshape(4, 4).T, (4, 16), (16, 4)),
        (np.arange(16, dtype=np.int32)[2:], (4,), (4,)),
        (np.arange(16, dtype=np.int32)[2::2], (8,), (4,)),
    )
    def test_strides_c_contiguous(self, arr1, strides1, strides2):
        self.assertEqual(arr1.strides, strides1)
        sharr = SharedNdarray.from_array(arr1, c_contiguous=True)
        arr2 = sharr.getarray()
        self.assertEqual(arr2.strides, strides2)
        np.testing.assert_array_equal(arr1, arr2)
        del arr2  # fixes `BufferError: cannot close exported pointers exist` in `SharedMemory.__del__`

    @parametrize(
        (np.arange(16, dtype=np.int32).reshape(4, 4).T, (4, 16)),
        (np.arange(16, dtype=np.int32)[2:], (4,)),
        (np.arange(16, dtype=np.int32)[2::2], (8,)),
    )
    def test_strides_non_contiguous(self, arr1, strides):
        self.assertEqual(arr1.strides, strides)
        sharr = SharedNdarray.from_array(arr1, c_contiguous=False)
        arr2 = sharr.getarray()
        self.assertEqual(arr2.strides, strides)
        np.testing.assert_array_equal(arr1, arr2)
        del arr2  # fixes `BufferError: cannot close exported pointers exist` in `SharedMemory.__del__`

    @parametrize(
        (np.arange(16, dtype=np.int32),),
        (np.arange(16, dtype=np.int32).reshape(4, 4).T,),
        (np.arange(16, dtype=np.int32)[2:],),
        (np.arange(16, dtype=np.int32)[2::2],),
    )
    def test_pickle(self, arr):
        # c-contiguous
        sharr1 = SharedNdarray.from_array(arr, True)
        sharr2 = pickle.loads(pickle.dumps(sharr1))  # nosec

        self.assertEqual(sharr1.tobytes(), sharr2.tobytes())
        np.testing.assert_array_equal(sharr1.getarray(), sharr2.getarray())

        # non-contiguous
        sharr1 = SharedNdarray.from_array(arr, False)
        sharr2 = pickle.loads(pickle.dumps(sharr1))  # nosec

        self.assertEqual(sharr1.tobytes(), sharr2.tobytes())
        np.testing.assert_array_equal(sharr1.getarray(), sharr2.getarray())

    @parametrize(
        (np.arange(16, dtype=np.int32),),
        (np.arange(16, dtype=np.int32).reshape(4, 4).T,),
        (np.arange(16, dtype=np.int32)[2:],),
        (np.arange(16, dtype=np.int32)[2::2],),
    )
    def test_properties(self, arr):
        # c-contiguous
        sharr = SharedNdarray.from_array(arr, True)
        self.assertEqual(arr.shape, sharr.shape)
        self.assertEqual(arr.dtype, sharr.dtype)
        self.assertEqual(arr.ndim, sharr.ndim)
        self.assertIsNone(sharr.strides)
        self.assertEqual(arr.nbytes, sharr.nbytes)
        self.assertEqual(arr.size, sharr.size)
        self.assertEqual(arr.nbytes, sharr.rawsize)

        # non-contiguous
        sharr = SharedNdarray.from_array(arr, False)
        self.assertEqual(arr.shape, sharr.shape)
        self.assertEqual(arr.dtype, sharr.dtype)
        self.assertEqual(arr.ndim, sharr.ndim)
        if sharr.strides is not None:
            self.assertEqual(arr.strides, sharr.strides)
        self.assertEqual(arr.size, sharr.size)
        self.assertEqual(arr.nbytes, sharr.nbytes)


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
