"""Hour-angle sampling and coverage in seconds and radians (MADIV-28 T1.3)."""

from __future__ import annotations

import math

import astropy.units as U
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from almasim.services.instrument.constants import (
    ALMA_ELEVATION_M,
    ALMA_LATITUDE_DEG,
    ALMA_LONGITUDE_DEG,
)
from almasim.services.interferometry.observing import (
    ELEVATION_LIMIT_DEG,
    HOUR_ANGLE_LIMIT_H,
    SCAN_LENGTH_S,
    TRACK_SPAN_FACTOR,
    Track,
    elevation_rad,
    execution_track,
    hour_angle_coverage,
    hour_angle_samples,
    track_start_mjd_s,
)

pytestmark = pytest.mark.unit

ALMA = EarthLocation.from_geodetic(
    lon=ALMA_LONGITUDE_DEG * U.deg, lat=ALMA_LATITUDE_DEG * U.deg, height=ALMA_ELEVATION_M * U.m
)
SIDEREAL_RATE = 2 * math.pi / 86164.0905  # rad per second of time
LAT = math.radians(ALMA_LATITUDE_DEG)


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


def test_a_track_spreads_its_scans_evenly_over_its_span():
    track = Track(0.0, span_s=100.0, scans=3)
    offsets = track.sample_offsets_s(9, 6.0)
    # Three scans of three 6 s samples, 18 s each, with two gaps of (100 - 54) / 2.
    np.testing.assert_allclose(offsets, [0, 6, 12, 41, 47, 53, 82, 88, 94])
    assert offsets[-1] + 6.0 == pytest.approx(100.0), "the last scan ends at the span"


def test_a_track_shorter_than_its_samples_observes_them_back_to_back():
    offsets = Track(0.0, span_s=10.0, scans=4).sample_offsets_s(5, 6.0)
    np.testing.assert_allclose(offsets, np.arange(5) * 6.0)
    assert Track(0.0, 0.0, scans=9).sample_offsets_s(3, 6.0).size == 3, "at most a scan a sample"


@pytest.mark.parametrize("middle", [-0.6, 0.0, 0.3])
def test_a_tracks_hour_angles_are_centred_on_its_middle(middle):
    angles = Track(middle, span_s=3600.0, scans=4).hour_angles_rad(40, 6.0)
    assert 0.5 * (angles[0] + angles[-1]) == pytest.approx(middle)
    # The first sample's middle to the last's: the span less one sample.
    assert angles[-1] - angles[0] == pytest.approx((3600.0 - 6.0) * SIDEREAL_RATE)


def test_by_default_a_block_is_scans_of_about_the_scan_length_over_its_span():
    track = execution_track(1527.0, LAT, math.radians(-30.0), np.random.default_rng(0))
    assert track.scans == round(1527.0 / SCAN_LENGTH_S)
    assert track.span_s == pytest.approx(TRACK_SPAN_FACTOR * 1527.0)
    assert execution_track(20.0, LAT, 0.0, np.random.default_rng(0)).scans == 1


def test_given_values_set_the_track():
    track = execution_track(
        998.0, LAT, 0.0, np.random.default_rng(0), hour_angle_h=-2.17, span_s=3407.0, scans=10
    )
    assert track == Track(-2.17 * math.pi / 12, 3407.0, 10)
    assert execution_track(998.0, LAT, 0.0, np.random.default_rng(0), span_s=10.0).span_s == 998.0


def test_drawn_tracks_spread_over_the_hour_angles_above_the_elevation_limit():
    dec = math.radians(-30.0)
    middles = np.array(
        [
            execution_track(3000.0, LAT, dec, np.random.default_rng(seed)).middle_hour_angle_rad
            for seed in range(400)
        ]
    )
    limit = HOUR_ANGLE_LIMIT_H * math.pi / 12
    assert np.abs(middles).max() <= limit
    assert np.median(np.abs(middles)) == pytest.approx(limit / 2, rel=0.15)
    # A source that rises little is observed only where the whole track stays up.
    low = math.radians(20.0)
    for seed in range(50):
        track = execution_track(3000.0, LAT, low, np.random.default_rng(seed))
        angles = track.hour_angles_rad(100, 30.0)
        assert np.degrees(elevation_rad(angles, LAT, low)).min() >= ELEVATION_LIMIT_DEG - 1e-6


def test_a_source_never_above_the_limit_is_observed_at_transit():
    track = execution_track(600.0, LAT, math.radians(60.0), np.random.default_rng(3))
    assert track.middle_hour_angle_rad == 0.0


def test_a_tracks_mean_elevation_is_highest_at_transit():
    dec = math.radians(-30.0)
    at_transit = Track(0.0, 600.0).mean_elevation_deg(100, 6.0, LAT, dec)
    assert at_transit == pytest.approx(90.0 - abs(ALMA_LATITUDE_DEG + 30.0), abs=0.1)
    assert Track(math.radians(30.0), 600.0).mean_elevation_deg(100, 6.0, LAT, dec) < at_transit
