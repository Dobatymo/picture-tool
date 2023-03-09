import numpy as np
from genutility.test import MyTestCase, parametrize

from picturetool.ml_utils import faiss_duplicates_threshold, faiss_from_array, faiss_to_pairs
from picturetool.utils import l2squared_duplicates_chunk, np_sorted, unique_pairs


class TestMlUtils(MyTestCase):
    @classmethod
    def setUpClass(cls):
        seed = 0
        rng = np.random.default_rng(seed)

        cls.inputs = {}
        cls.truths_3d = {}

        dims_3d = [1, 2, 3, 89]
        cls.arr = rng.uniform(0, 1, size=(max(dims_3d), 10)).astype(np.float32)

        for d in dims_3d:
            pairs = unique_pairs(l2squared_duplicates_chunk(cls.arr[None, :d, :], cls.arr[:d, None, :], threshold=1.1))
            cls.truths_3d[d] = np_sorted(pairs)

    def test_faiss_duplicates_threshold(self):
        arr = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        index = faiss_from_array(arr, "l2-squared")

        truth = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        pairs, dists = faiss_to_pairs(faiss_duplicates_threshold(index, 1000, 2.1))
        result = np_sorted(pairs)
        np.testing.assert_array_equal(result, truth)

        truth = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
        pairs, dists = faiss_to_pairs(faiss_duplicates_threshold(index, 1000, 1.1))
        result = np_sorted(pairs)
        np.testing.assert_array_equal(result, truth)

    @parametrize(
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
        (3, 3),
        (89, 23),
    )
    def test_faiss_duplicates_threshold_random(self, d, chunksize):
        index = faiss_from_array(self.arr[:d, :], "l2-squared")
        pairs, dists = faiss_to_pairs(faiss_duplicates_threshold(index, chunksize, 1.1))
        result = np_sorted(pairs)
        np.testing.assert_array_equal(result, self.truths_3d[d])


if __name__ == "__main__":
    import unittest

    unittest.main()
