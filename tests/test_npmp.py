import numpy as np
from genutility.test import MyTestCase, parametrize

from picturetool.npmp import ChunkedParallel, SharedNdarray


def sum_chunk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a + b, axis=-1)


class NpmpTest(MyTestCase):
    @classmethod
    def setUpClass(cls):
        seed = 0
        rng = np.random.default_rng(seed)
        cls.arr = rng.uniform(0, 1, size=(100, 100)).astype(np.float32)
        cls.truth = sum_chunk(cls.arr, cls.arr)
        cls.chunksize = 10

    @parametrize(
        (1,),
        (10,),
        (100,),
        (1000,),
    )
    def test_mp(self, chunksize):
        sharr = SharedNdarray.from_array(self.arr)
        it = ChunkedParallel(
            sum_chunk, sharr, sharr, (chunksize,), ordered=True, backend="multiprocessing", pass_coords=False
        )
        result = np.concatenate(list(it))
        np.testing.assert_allclose(result, self.truth)

    @parametrize(
        (1,),
        (10,),
        (100,),
        (1000,),
    )
    def test_mt(self, chunksize):
        it = ChunkedParallel(
            sum_chunk, self.arr, self.arr, (chunksize,), ordered=True, backend="threading", pass_coords=False
        )
        result = np.concatenate(list(it))
        np.testing.assert_allclose(result, self.truth)


if __name__ == "__main__":
    import unittest

    unittest.main()
