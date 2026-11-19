"""The residual bandpass ``noise fit`` measures across a window."""

from __future__ import annotations

import numpy as np
import pytest

from almasim.services.interferometry.corruptions import draw_bandpass_ripple
from almasim.services.noise.bandpass_fit import BandpassSums, fit_bandpass

NANT, NCHAN = 12, 256
FREQ = 40e9 + 1e6 * np.arange(NCHAN)  # band 1, 1 MHz channels


def _sums(amp: float, phase_deg: float, sigma: float, seed: int = 0) -> BandpassSums:
    """A bright point source on every baseline, its antennas' ripple of 25 MHz."""
    rng = np.random.default_rng(seed)
    ripple = draw_bandpass_ripple(
        ((12.0, 50.0, 25.0, amp, phase_deg),), np.full(NANT, 12.0), rng
    ).factors(FREQ)
    a1, a2 = np.triu_indices(NANT, 1)
    sums = BandpassSums(NANT, FREQ)
    model = np.full((a1.size, NCHAN), 1.0 + 0j)
    w = np.full((a1.size, NCHAN, 2), 1.0 / sigma**2)  # WEIGHT: 1 / the noise variance
    for _ in range(20):  # integrations, each with its own gain phase
        gain = np.exp(1j * rng.uniform(-1, 1, NANT))
        signal = (gain[a1] * np.conj(gain[a2]))[:, None] * ripple[a1] * np.conj(ripple[a2])
        noise = sigma * (
            rng.standard_normal((a1.size, NCHAN, 2)) + 1j * rng.standard_normal((a1.size, NCHAN, 2))
        )
        sums.add(w, signal[:, :, None] * model[:, :, None] + noise, model, a1, a2)
    return sums


@pytest.mark.unit
def test_a_ripple_is_measured_with_its_size_and_period():
    found, why = fit_bandpass(_sums(0.02, 3.0, sigma=0.3), 1.0, np.random.default_rng(1))
    assert why is None and found is not None
    assert (found.chunks, found.chunk_mhz, found.antennas) == (64, 4.0, NANT)
    assert found.amp_rms == pytest.approx(0.02 / np.sqrt(2), rel=0.25)
    assert found.phase_rms_deg == pytest.approx(3.0 / np.sqrt(2), rel=0.25)
    assert found.period_mhz == pytest.approx(25.0, rel=0.1)


@pytest.mark.unit
def test_noise_alone_leaves_no_ripple():
    found, _ = fit_bandpass(_sums(0.0, 0.0, sigma=0.3), 1.0, np.random.default_rng(1))
    assert found is not None
    assert found.amp_rms < 0.3 * found.thermal_amp_rms
    assert found.phase_rms_deg < 0.3 * found.thermal_phase_rms_deg


@pytest.mark.unit
def test_a_window_of_few_wide_channels_cannot_show_a_ripple():
    sums = BandpassSums(NANT, 230e9 + 15.625e6 * np.arange(8))
    found, why = fit_bandpass(sums, 1.0, np.random.default_rng(1))
    assert found is None and "too few" in why


@pytest.mark.unit
def test_a_ripple_is_found_where_no_single_baseline_s_chunk_is_bright():
    # S/N about 2 per (baseline, chunk) over the observation: the antennas pool it.
    found, why = fit_bandpass(_sums(0.05, 3.0, sigma=3.0, seed=4), 1.0, np.random.default_rng(5))
    assert why is None and found is not None
    assert found.thermal_amp_rms > 0.02
    assert found.amp_rms == pytest.approx(0.05 / np.sqrt(2), rel=0.35)


@pytest.mark.unit
def test_a_window_too_faint_for_a_ripple_says_so():
    found, why = fit_bandpass(_sums(0.01, 1.0, sigma=30.0), 1.0, np.random.default_rng(1))
    assert found is None and "too faint" in why
