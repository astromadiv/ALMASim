"""The ALMA primary-beam model (MADIV-28 T4.5).

CASA's model for ALMA (its ``airy`` voltage pattern): an Airy disk of a
10.7 m aperture with a 0.75 m central blockage for the 12 m antennas, 6.25 m
and 0.75 m for the 7 m ones. Any other diameter is an unblocked Airy disk of
that diameter. The power response at angle ``theta`` and wavelength
``lambda``, with ``x = pi D theta / lambda`` and ``e`` the blockage over the
aperture, is::

    [(jinc(x) - e^2 jinc(e x)) / (1 - e^2)]^2,   jinc(z) = 2 J1(z) / z

Its half-power width is 1.15 lambda / D_dish (12 m) and 1.14 (7 m), close
to the 1.13 ALMA quotes for the measured beams. The simulator, the
total-power simulation and the ``--pbcor`` correction all use it.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S

# Nominal dish diameter -> (effective aperture, central blockage) in metres.
ALMA_AIRY_APERTURES = {12.0: (10.7, 0.75), 7.0: (6.25, 0.75)}


# The field-of-view factor (x lambda / D) imaging parameter files used before
# MADIV-28 T2.1; still recognised when their dish diameter is read back.
LEGACY_FOV_FACTOR = 1.12


def airy_aperture(dish_diameter: float) -> tuple[float, float]:
    """(effective aperture, blockage) in metres for a nominal dish diameter."""
    for nominal, aperture in ALMA_AIRY_APERTURES.items():
        if abs(dish_diameter - nominal) < 0.5:
            return aperture
    return float(dish_diameter), 0.0


def _jinc(z: np.ndarray) -> np.ndarray:
    from scipy.special import j1

    z = np.asarray(z, dtype=float)
    safe = np.where(z == 0, 1.0, z)
    return np.where(z == 0, 1.0, 2.0 * j1(safe) / safe)


def primary_beam_voltage(
    radius_rad: np.ndarray, frequency_hz: float, dish_diameter: float
) -> np.ndarray:
    """Voltage response (1 on axis) at ``radius_rad`` from the pointing centre:
    the blocked aperture's Airy pattern, negative in its odd sidelobes."""
    aperture, blockage = airy_aperture(dish_diameter)
    x = math.pi * aperture * np.asarray(radius_rad, dtype=float) * frequency_hz / SPEED_OF_LIGHT_M_S
    e = blockage / aperture
    return (_jinc(x) - e**2 * _jinc(e * x)) / (1.0 - e**2)


def primary_beam_response(
    radius_rad: np.ndarray, frequency_hz: float, dish_diameter: float
) -> np.ndarray:
    """Power response (1 on axis) at ``radius_rad`` from the pointing centre."""
    return primary_beam_voltage(radius_rad, frequency_hz, dish_diameter) ** 2


def baseline_beam_response(
    radius_rad: np.ndarray, frequency_hz: float, diameter_1: float, diameter_2: float
) -> np.ndarray:
    """The response of a baseline between two dishes: the product of their
    voltage patterns, each dish's power response when the two are alike."""
    if abs(float(diameter_1) - float(diameter_2)) < 1e-9:
        return primary_beam_response(radius_rad, frequency_hz, diameter_1)
    return primary_beam_voltage(radius_rad, frequency_hz, diameter_1) * primary_beam_voltage(
        radius_rad, frequency_hz, diameter_2
    )


def _half_power_radius(dish_diameter: float, frequency_hz: float) -> float:
    """Radius (rad) where the response falls to 0.5, by bisection."""
    aperture, _ = airy_aperture(dish_diameter)
    low, high = 0.0, 3.0 * SPEED_OF_LIGHT_M_S / frequency_hz / aperture
    for _ in range(60):
        middle = 0.5 * (low + high)
        if primary_beam_response(np.array(middle), frequency_hz, dish_diameter) > 0.5:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@cache
def airy_fwhm_factor(dish_diameter: float) -> float:
    """Half-power full width of this module's beam model, in units of lambda / D.

    About 1.15 for the 12 m antennas and 1.14 for the 7 m ones (blocked
    apertures), 1.03 for an unblocked dish of any other diameter. It is the
    one primary-beam width ALMASim uses: imaging field of view, the
    simulator's primary beam and the total-power beam (MADIV-28 T2.1).
    """
    frequency = 100e9
    wavelength = SPEED_OF_LIGHT_M_S / frequency
    return (
        2.0
        * _half_power_radius(float(dish_diameter), frequency)
        * float(dish_diameter)
        / wavelength
    )


def primary_beam_fwhm_rad(dish_diameter: float, frequency_hz: Any) -> Any:
    """Half-power full width (rad) of the primary beam at ``frequency_hz``."""
    wavelength = SPEED_OF_LIGHT_M_S / np.asarray(frequency_hz, dtype=float)
    return airy_fwhm_factor(float(dish_diameter)) * wavelength / float(dish_diameter)


def dish_diameter_from_fov(
    fov_arcsec: float, reference_frequency_hz: float, fov_factor: float | None = None
) -> float:
    """Dish diameter (m) behind a ``fov_per_frequency`` of ``fov_factor`` lambda / D.

    Without ``fov_factor`` the field of view is the beam model's half-power
    width (:func:`primary_beam_fwhm_rad`): a nominal ALMA dish is returned
    when it matches, also for parameter files written before MADIV-28 T2.1
    with ``LEGACY_FOV_FACTOR``; otherwise the unblocked dish that would.
    """
    if fov_arcsec <= 0 or reference_frequency_hz <= 0:
        raise ValueError("the field of view and the frequency must be positive")
    fov_rad = math.radians(fov_arcsec / 3600.0)
    wavelength = SPEED_OF_LIGHT_M_S / reference_frequency_hz
    if fov_factor is not None:
        return fov_factor * wavelength / fov_rad
    for nominal in ALMA_AIRY_APERTURES:
        for factor in (airy_fwhm_factor(nominal), LEGACY_FOV_FACTOR):
            if math.isclose(factor * wavelength / nominal, fov_rad, rel_tol=1e-9):
                return nominal
    return airy_fwhm_factor(1.0) * wavelength / fov_rad
