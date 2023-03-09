import numpy as np
from genutility.test import MyTestCase

from picturetool.ml_utils import faiss_duplicates_threshold, faiss_from_array, faiss_to_pairs


class TestMlUtils(MyTestCase):
    def test_faiss_duplicates_threshold(self):
        arr = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        index = faiss_from_array(arr, "l2-squared")

        truth = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        pairs, dists = faiss_to_pairs(faiss_duplicates_threshold(index, 1000, 2.1))
        result = set(map(tuple, pairs))
        self.assertEqual(truth, result)

        truth = {(0, 1), (0, 2), (1, 3), (2, 3)}
        pairs, dists = faiss_to_pairs(faiss_duplicates_threshold(index, 1000, 1.1))
        result = set(map(tuple, pairs))
        self.assertEqual(truth, result)


if __name__ == "__main__":
    import unittest

    unittest.main()
