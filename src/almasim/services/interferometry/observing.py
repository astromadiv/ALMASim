"""When and how often the simulated array observes (MADIV-28 T1.3).

The legacy simulator took the integration time (in hours) as seconds when it
built the hour-angle track, and then fed that span, in degrees, to
``sin``/``cos`` as radians: every track came out about 63 times too short
(a 3.9 h observation spanned 0.016 rad instead of 1.02 rad), close to a
snapshot. These helpers take seconds and return radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import astropy.units as U
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ..instrument.constants import ALMA_ELEVATION_M, ALMA_LATITUDE_DEG, ALMA_LONGITUDE_DEG

# Scan lengths (s) tried in turn until the track has at most MAX_EPOCHS samples.
SCAN_TIMES_S = (6.0, 8.064, 18.144, 30.24)
MAX_EPOCHS = 200


def exact_samples(duration_s: float, sample_time_s: float) -> tuple[int, float]:
    """(number of samples, sample time in s) that observe ``duration_s`` in
    full: the whole number of samples nearest to ``duration_s / sample_time_s``
    (at least one), each lasting ``duration_s`` divided by that number. The
    samples add up to the time on source, which a short observation would
    otherwise gain or lose a large share of (16 s in 6 s scans)."""
    duration_s = float(duration_s)
    samples = max(1, round(duration_s / float(sample_time_s)))
    return samples, duration_s / samples if duration_s > 0 else float(sample_time_s)


def hour_angle_samples(integration_time_s: float) -> tuple[int, float]:
    """(number of epochs, scan time in s) for an integration time in seconds.

    The shortest scan that keeps the track at ``MAX_EPOCHS`` samples or fewer
    is used; a longer integration keeps the longest scan. The epochs observe
    the integration time in full (``exact_samples``), so the scan time is
    within half a scan of the length chosen.
    """
    for scan_time in SCAN_TIMES_S:
        n_epochs, exact = exact_samples(integration_time_s, scan_time)
        if n_epochs <= MAX_EPOCHS:
            break
    return n_epochs, exact


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


# An execution block's track, as the 70 MADIV-16 windows of the noise study
# (MADIV-28) observed theirs: its middle hour angle within +-HOUR_ANGLE_LIMIT_H
# (|HA| median 1.24 h, uniform over +-2.5 h gives 1.25 h), its time on source
# in scans of about SCAN_LENGTH_S (median 167 s between gaps) spread over
# TRACK_SPAN_FACTOR times that time (median 1.27: the calibrators' scans and
# the other fields' take the rest), the source above ELEVATION_LIMIT_DEG.
HOUR_ANGLE_LIMIT_H = 2.5
SCAN_LENGTH_S = 170.0
TRACK_SPAN_FACTOR = 1.27
ELEVATION_LIMIT_DEG = 30.0


@dataclass(frozen=True)
class Track:
    """When one execution block observes its source: ``scans`` equal scans, the
    first starting and the last ending ``span_s`` apart, around the hour angle
    ``middle_hour_angle_rad``."""

    middle_hour_angle_rad: float
    span_s: float
    scans: int = 1

    def sample_offsets_s(self, samples: int, sample_time_s: float) -> np.ndarray:
        """Each sample's start, in seconds from the first's: the samples split
        into the scans as evenly as they go, the scans spread evenly over the
        span (back to back when the span is shorter than the samples)."""
        samples = max(1, int(samples))
        scans = max(1, min(int(self.scans), samples))
        counts = np.diff(np.linspace(0, samples, scans + 1).round().astype(int))
        observed = samples * float(sample_time_s)
        gap = max(0.0, float(self.span_s) - observed) / (scans - 1) if scans > 1 else 0.0
        starts = np.concatenate(([0], np.cumsum(counts)[:-1])) * float(sample_time_s)
        starts = starts + np.arange(scans) * gap
        within = np.concatenate([np.arange(n) for n in counts]) * float(sample_time_s)
        return np.repeat(starts, counts) + within

    def hour_angles_rad(self, samples: int, sample_time_s: float) -> np.ndarray:
        """Each sample's hour angle (rad) at its middle, the track centred on
        ``middle_hour_angle_rad``."""
        offsets = self.sample_offsets_s(samples, sample_time_s) + 0.5 * float(sample_time_s)
        middle = 0.5 * (offsets[0] + offsets[-1])
        return self.middle_hour_angle_rad + (offsets - middle) * (2.0 * math.pi / SIDEREAL_DAY_S)

    def mean_elevation_deg(
        self, samples: int, sample_time_s: float, lat_rad: float, dec_rad: float
    ) -> float:
        """The source's mean elevation over the track's samples (degrees)."""
        angles = self.hour_angles_rad(samples, sample_time_s)
        return float(np.degrees(np.mean(elevation_rad(angles, lat_rad, dec_rad))))


def elevation_rad(hour_angle_rad: np.ndarray | float, lat_rad: float, dec_rad: float) -> np.ndarray:
    """The source's elevation (rad) at ``hour_angle_rad``."""
    sin_el = math.sin(lat_rad) * math.sin(dec_rad) + math.cos(lat_rad) * math.cos(dec_rad) * np.cos(
        hour_angle_rad
    )
    return np.arcsin(np.clip(sin_el, -1.0, 1.0))


def execution_track(
    on_source_s: float,
    lat_rad: float,
    dec_rad: float,
    rng: np.random.Generator,
    *,
    hour_angle_h: float | None = None,
    span_s: float | None = None,
    scans: int | None = None,
) -> Track:
    """An execution block's track for ``on_source_s`` on source.

    ``scans`` defaults to the time on source in scans of about SCAN_LENGTH_S,
    ``span_s`` to TRACK_SPAN_FACTOR times the time on source; ``hour_angle_h``
    (the track's middle) is drawn with ``rng``, uniformly over the hour angles
    within +-HOUR_ANGLE_LIMIT_H at which the whole track stays above
    ELEVATION_LIMIT_DEG, or transit when there are none.
    """
    on_source_s = max(float(on_source_s), 0.0)
    if scans is None:
        scans = max(1, round(on_source_s / SCAN_LENGTH_S))
    if span_s is None:
        span_s = TRACK_SPAN_FACTOR * on_source_s
    span_s = max(float(span_s), on_source_s)
    if hour_angle_h is not None:
        return Track(float(hour_angle_h) * math.pi / 12.0, span_s, int(scans))
    half_span = 0.5 * span_s * 2.0 * math.pi / SIDEREAL_DAY_S
    cos_limit = (
        math.sin(math.radians(ELEVATION_LIMIT_DEG)) - math.sin(lat_rad) * math.sin(dec_rad)
    ) / (math.cos(lat_rad) * math.cos(dec_rad))
    above = math.pi if cos_limit <= -1.0 else (math.acos(cos_limit) if cos_limit < 1.0 else 0.0)
    reach = min(HOUR_ANGLE_LIMIT_H * math.pi / 12.0, above - half_span)
    middle = float(rng.uniform(-reach, reach)) if reach > 0.0 else 0.0
    return Track(middle, span_s, int(scans))


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
