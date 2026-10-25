"""ALMA's channels from the archive's resolution (MADIV-28)."""

import numpy as np
import pytest

from almasim.services.instrument.channels import (
    HANNING_RHO,
    Channels,
    averaged_correlation,
    channels_from_resolution,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("resolution_khz", "bandwidth_mhz", "width_khz", "averaged", "noise_over_width"),
    [
        # The MADIV-16 windows' RESOLUTION, CHAN_WIDTH and EFFECTIVE_BW, by class.
        (31250.00, 1875.0, 15625.0, 1, 2.667),
        (122.07, 62.5, 61.035, 1, 2.667),
        (2257.81, 937.5, 1953.125, 2, 1.600),
        (1938.48, 1875.0, 1953.125, 4, 1.231),
        (3904.30, 1875.0, 3906.25, 8, 1.104),
        (7812.01, 1875.0, 7812.5, 16, 1.049),
    ],
)
def test_resolution_gives_the_width_and_the_averaging(
    resolution_khz, bandwidth_mhz, width_khz, averaged, noise_over_width
):
    channels = channels_from_resolution(resolution_khz * 1e3, bandwidth_mhz * 1e6)
    assert channels.width_hz == pytest.approx(width_khz * 1e3, rel=1e-5)
    assert channels.averaged == averaged
    assert channels.noise_bandwidth_hz / channels.width_hz == pytest.approx(
        noise_over_width, abs=2e-3
    )


def test_a_resolution_off_the_grid_is_the_width():
    assert channels_from_resolution(1e6, 2e9) == Channels(1e6)


def test_averaging_dilutes_the_hanning_correlation():
    assert averaged_correlation(1) == pytest.approx(HANNING_RHO)
    # The adjacent-channel noise correlation the MADIV-16 windows show for each N.
    measured = {2: 0.30, 4: 0.115, 8: 0.052, 16: 0.025}
    for n, rho1 in measured.items():
        got1, got2 = averaged_correlation(n)
        assert got1 == pytest.approx(rho1, abs=0.005)
        assert got2 == 0.0


def test_averaged_correlation_matches_averaged_noise():
    rng = np.random.default_rng(0)
    native = np.convolve(rng.standard_normal(400_002), [0.25, 0.5, 0.25], mode="valid")
    channels = native[: len(native) // 4 * 4].reshape(-1, 4).mean(axis=1)
    measured = [np.corrcoef(channels[:-k], channels[k:])[0, 1] for k in (1, 2)]
    assert measured == pytest.approx(averaged_correlation(4), abs=0.01)
