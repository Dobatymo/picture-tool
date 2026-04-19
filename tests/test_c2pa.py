from unittest import SkipTest

from genutility.test import MyTestCase

try:
    from picturetool.c2pa_utils import C2paError, c2pa_json
except ModuleNotFoundError:
    raise SkipTest("Skipping c2pa tests since c2pa-python is not available")


class TestC2pa(MyTestCase):
    def test_c2pa_json_no_manifest(self):
        with self.assertRaises(C2paError.ManifestNotFound):
            c2pa_json("c2pa-org_public-testfiles/adobe-20220124-A.jpg")

    def test_c2pa_json_good_manifest(self):
        validation_state = "Valid"
        result = c2pa_json("c2pa-org_public-testfiles/adobe-20220124-C.jpg")
        self.assertEqual(validation_state, result["validation_state"])

    def test_c2pa_json_bad_manifest(self):
        validation_state = "Invalid"
        result = c2pa_json("c2pa-org_public-testfiles/adobe-20220124-C-mod.jpg")
        self.assertEqual(validation_state, result["validation_state"])


if __name__ == "__main__":
    import unittest

    unittest.main()
