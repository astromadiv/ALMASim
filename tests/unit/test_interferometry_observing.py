"""Hour-angle sampling and coverage in seconds and radians (MADIV-28 T1.3)."""

from __future__ import annotations

import math

import astropy.units as U
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from almasim.services.instrument.constants import (
    ALMA_ELEVATION_M,
    ALMA_LATITUDE_DEG,
    ALMA_LONGITUDE_DEG,
)
from almasim.services.interferometry.observing import (
    hour_angle_coverage,
    hour_angle_samples,
    track_start_mjd_s,
)

pytestmark = pytest.mark.unit

ALMA = EarthLocation.from_geodetic(
    lon=ALMA_LONGITUDE_DEG * U.deg, lat=ALMA_LATITUDE_DEG * U.deg, height=ALMA_ELEVATION_M * U.m
)
SIDEREAL_RATE = 2 * math.pi / 86164.0905  # rad per second of time


@pytest.mark.parametrize(
    ("integration_s", "expected"),
    [
        (450.0, (75, 6.0)),
        (1200.0, (200, 6.0)),
        (1500.0, (186, 8.064)),
        (13970.88, (462, 30.24)),
        (1.0, (1, 6.0)),
    ],
)
def test_one_epoch_per_scan_in_seconds(integration_s, expected):
    assert hour_angle_samples(integration_s) == expected


@pytest.mark.parametrize("ra_deg", [0.0, 36.6, 180.0, 359.9])
def test_a_one_hour_track_spans_about_a_quarter_radian(ra_deg):
    start, end = hour_angle_coverage("2017-03-14", 3600.0, math.radians(ra_deg), ALMA)
    assert end - start == pytest.approx(3600.0 * SIDEREAL_RATE, rel=1e-3)
    assert start == pytest.approx(-end, rel=1e-3), "centred on the middle of the track"
    assert abs(start) < math.pi and abs(end) < math.pi


def _hour_angle(mjd_s: float, ra_rad: float) -> float:
    lst = Time(mjd_s / 86400.0, format="mjd", scale="utc").sidereal_time("apparent", ALMA.lon)
    return math.remainder(float(lst.rad) - ra_rad, 2 * math.pi)


@pytest.mark.parametrize("ra_deg", [0.0, 36.6, 180.0, 359.9])
@pytest.mark.parametrize("hour_angle", [-1.0, 0.0, 0.4])
def test_a_track_starts_at_its_hour_angle_from_transit(ra_deg, hour_angle):
    when = track_start_mjd_s("2017-03-14", math.radians(ra_deg), hour_angle, ALMA)
    assert _hour_angle(when, math.radians(ra_deg)) == pytest.approx(hour_angle, abs=1e-4)
    midnight = Time("2017-03-14T00:00:00", scale="utc").mjd * 86400.0
    assert midnight - 86400.0 < when < midnight + 2 * 86400.0, "around the observation date"


def test_a_source_at_transit_is_at_its_highest():
    ra, dec = math.radians(83.8), math.radians(-5.4)
    when = Time(track_start_mjd_s("2017-03-14", ra, 0.0, ALMA) / 86400.0, format="mjd")
    altaz = SkyCoord(ra * U.rad, dec * U.rad).transform_to(AltAz(obstime=when, location=ALMA))
    assert altaz.alt.deg == pytest.approx(
        90.0 - abs(ALMA_LATITUDE_DEG - math.degrees(dec)), abs=0.5
    )
