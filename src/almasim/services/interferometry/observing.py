"""When and how often the simulated array observes (MADIV-28 T1.3).

The legacy simulator took the integration time (in hours) as seconds when it
built the hour-angle track, and then fed that span, in degrees, to
``sin``/``cos`` as radians: every track came out about 63 times too short
(a 3.9 h observation spanned 0.016 rad instead of 1.02 rad), close to a
snapshot. These helpers take seconds and return radians.
"""

from __future__ import annotations

import math

import astropy.units as U
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ..instrument.constants import ALMA_ELEVATION_M, ALMA_LATITUDE_DEG, ALMA_LONGITUDE_DEG

# Scan lengths (s) tried in turn until the track has at most MAX_EPOCHS samples.
SCAN_TIMES_S = (6.0, 8.064, 18.144, 30.24)
MAX_EPOCHS = 200


def hour_angle_samples(integration_time_s: float) -> tuple[int, float]:
    """(number of epochs, scan time in s) for an integration time in seconds.

    The shortest scan that keeps the track at ``MAX_EPOCHS`` samples or fewer
    is used; a longer integration keeps the longest scan. At least one epoch.
    """
    for scan_time in SCAN_TIMES_S:
        n_epochs = int(float(integration_time_s) / scan_time)
        if n_epochs <= MAX_EPOCHS:
            break
    return max(1, n_epochs), scan_time


def _wrap(angle_rad: float) -> float:
    """``angle_rad`` in (-pi, pi]."""
    wrapped = math.remainder(angle_rad, 2.0 * math.pi)
    return math.pi if wrapped == -math.pi else wrapped


def hour_angle_coverage(
    observation_date: str,
    integration_time_s: float,
    ra_rad: float,
    location: EarthLocation,
) -> list[float]:
    """[start, end] hour angle (rad) of the track, relative to its middle.

    The observation starts at 00:00 UTC on ``observation_date`` and lasts
    ``integration_time_s``; the track is centred on its middle hour angle,
    so a one-hour observation spans about +-0.131 rad.
    """
    start = Time(observation_date + "T00:00:00", format="isot", scale="utc")
    duration = float(integration_time_s) * U.s
    middle, end = start + duration / 2, start + duration

    def hour_angle(time: Time) -> float:
        lst = time.sidereal_time("apparent", longitude=location.lon)
        return float(lst.rad) - ra_rad

    ha_middle = hour_angle(middle)
    return [_wrap(hour_angle(start) - ha_middle), _wrap(hour_angle(end) - ha_middle)]


# Seconds in a sidereal day: the hour angle advances 2 pi in one.
SIDEREAL_DAY_S = 86164.0905


def track_start_mjd_s(
    observation_date: str, ra_rad: float, hour_angle_rad: float, location: EarthLocation
) -> float:
    """When (MJD seconds, UTC) the source at ``ra_rad`` is at ``hour_angle_rad``
    around its transit at ``location`` on ``observation_date``."""
    midnight = Time(observation_date + "T00:00:00", format="isot", scale="utc")
    lst = float(midnight.sidereal_time("apparent", longitude=location.lon).rad)
    to_transit = (float(ra_rad) - lst) % (2.0 * math.pi)
    turns = (to_transit + float(hour_angle_rad)) / (2.0 * math.pi)
    return midnight.mjd * 86400.0 + turns * SIDEREAL_DAY_S


def alma_location() -> EarthLocation:
    """The ALMA site."""
    return EarthLocation.from_geodetic(
        lon=ALMA_LONGITUDE_DEG * U.deg,
        lat=ALMA_LATITUDE_DEG * U.deg,
        height=ALMA_ELEVATION_M * U.m,
    )


def clip_to_horizon(coverage: list, lat_rad: float, dec_rad: float) -> list:
    """``coverage`` ([start, end] hour angles, rad) limited to when the source is up.

    Changed in place and returned (MADIV-28 T5.11: moved out of Interferometer).
    """
    cos_w = -np.tan(lat_rad) * np.tan(dec_rad)
    if np.abs(cos_w) < 1.0:
        h_horizon = np.arccos(cos_w)
    elif np.abs(lat_rad - dec_rad) > np.pi / 2.0:
        h_horizon = 0
    else:
        h_horizon = np.pi
    if h_horizon > 0.0:
        if coverage[0] < -h_horizon:
            coverage[0] = -h_horizon
        if coverage[1] > h_horizon:
            coverage[1] = h_horizon
    return coverage
