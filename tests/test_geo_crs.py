"""CRS derivation + UTM zone math (pure, always available)."""
import pytest

from drishti.geo.crs import (
    DerivedCrs,
    derive_crs_from_track,
    haversine_m,
    utm_epsg,
    utm_zone_from_lon,
)


def test_utm_zone_from_lon_known():
    assert utm_zone_from_lon(0.0) == 31       # prime meridian
    assert utm_zone_from_lon(-123.1) == 10    # Vancouver
    assert utm_zone_from_lon(77.2) == 43      # Delhi
    assert utm_zone_from_lon(151.2) == 56     # Sydney


def test_utm_zone_bounds():
    for lon in (-180.0, -179.9, 179.9, 180.0):
        assert 1 <= utm_zone_from_lon(lon) <= 60


def test_utm_epsg_hemisphere():
    assert utm_epsg(77.2, 28.6) == 32643      # Delhi, north -> 326xx
    assert utm_epsg(151.2, -33.8) == 32756    # Sydney, south -> 327xx


def test_derive_from_track_median():
    lons = [77.10, 77.20, 77.30]
    lats = [28.50, 28.60, 28.70]
    crs = derive_crs_from_track(lons, lats)
    assert isinstance(crs, DerivedCrs)
    assert crs.epsg == 32643
    assert crs.zone == 43
    assert crs.hemisphere == "N"
    assert crs.derived_from == "gps_median_lonlat"


def test_derive_from_track_override():
    crs = derive_crs_from_track([77.0], [28.0], override_epsg=32633)
    assert crs.epsg == 32633
    assert crs.derived_from == "user_set"


def test_derive_from_track_errors():
    with pytest.raises(ValueError):
        derive_crs_from_track([], [])
    with pytest.raises(ValueError):
        derive_crs_from_track([1.0, 2.0], [1.0])          # mismatched length
    with pytest.raises(ValueError):
        derive_crs_from_track([200.0], [10.0])            # out-of-range lon


def test_haversine_known_distances():
    assert haversine_m(0, 0, 0, 0) == 0.0
    # one degree of latitude ~ 111.19 km on a sphere of r=6371008.8
    d = haversine_m(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111195.0, abs=5.0)
    # symmetry
    assert haversine_m(28.6, 77.2, 28.7, 77.3) == pytest.approx(
        haversine_m(28.7, 77.3, 28.6, 77.2)
    )
