from fractions import Fraction
from types import SimpleNamespace
from unittest import TestCase

from picturetool.viewer_gui import PictureWindow, gps_dms_to_dd


class ViewerGuiTest(TestCase):
    def test_gps_hemispheres(self):
        for lat_ref, lon_ref, lat_sign, lon_sign in [("N", "E", 1, 1), ("S", "W", -1, -1)]:
            with self.subTest(lat_ref=lat_ref, lon_ref=lon_ref):
                meta = {
                    "gps-lat": (Fraction(0), Fraction(30), Fraction(0)),
                    "gps-lat-ref": lat_ref,
                    "gps-lon": (Fraction(151), Fraction(12), Fraction(0)),
                    "gps-lon-ref": lon_ref,
                }
                window = SimpleNamespace(loaded={"meta": meta})
                lat, lon = PictureWindow._gps(window)
                self.assertAlmostEqual(gps_dms_to_dd(lat), lat_sign * 0.5)
                self.assertAlmostEqual(gps_dms_to_dd(lon), lon_sign * 151.2)
