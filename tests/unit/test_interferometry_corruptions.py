"""Visibility corruptions besides white thermal noise (MADIV-28)."""

import numpy as np
import pytest

from almasim.services.interferometry.corruptions import (
    AntennaCorruptions,
    CorruptionModel,
    antenna_ripple,
    baseline_antennas,
    correlated_noise,
    draw_antenna_corruptions,
    scan_index,
    spectral_kernel,
    turbulent_screen,
)
from almasim.services.interferometry.simulator import ChannelImagingConfig, image_configured_channel


@pytest.mark.unit
def test_empty_model_is_none_and_unknown_fields_fail():
    assert CorruptionModel.from_mapping(None) is None
    assert CorruptionModel.from_mapping({}) is None
    model = CorruptionModel.from_mapping({"gain_amp_rms": 0.1})
    assert model.has_antenna_errors and not model.correlates_channels
    with pytest.raises(ValueError, match="unknown"):
        CorruptionModel.from_mapping({"gain_amplitude": 0.1})


@pytest.mark.unit
def test_hanning_kernel_and_correlated_noise():
    kernel = spectral_kernel(2 / 3, 1 / 6)
    assert np.allclose(kernel / kernel.sum(), [0.25, 0.5, 0.25])
    assert np.isclose(np.sum(kernel**2), 1.0)
    shape = (200, 100)
    cube = np.stack([correlated_noise((7, 1), c, shape, 2.0, kernel) for c in range(6)])
    # Each channel keeps the requested noise; neighbours share a draw.
    assert np.isclose(cube.real.std(), 2.0, rtol=0.02)
    rho1 = np.mean(
        [np.corrcoef(cube[c].real.ravel(), cube[c + 1].real.ravel())[0, 1] for c in range(5)]
    )
    rho2 = np.mean(
        [np.corrcoef(cube[c].real.ravel(), cube[c + 2].real.ravel())[0, 1] for c in range(4)]
    )
    rho3 = np.corrcoef(cube[0].real.ravel(), cube[3].real.ravel())[0, 1]
    assert rho1 == pytest.approx(2 / 3, abs=0.02)
    assert rho2 == pytest.approx(1 / 6, abs=0.02)
    assert abs(rho3) < 0.02
    # The same channel and entropy give the same noise, in any task.
    assert np.array_equal(correlated_noise((7, 1), 3, shape, 2.0, kernel), cube[3])


@pytest.mark.unit
def test_kernel_matches_rho1_when_rho2_disagrees():
    kernel = spectral_kernel(0.3, 0.2)
    assert 2 * kernel[0] * kernel[1] == pytest.approx(0.3, abs=1e-6)
    with pytest.raises(ValueError):
        spectral_kernel(0.5, 0.4)


@pytest.mark.unit
@pytest.mark.parametrize("slope", [0.4, 5 / 6])
def test_screen_follows_its_power_law(slope):
    rng = np.random.default_rng(3)
    ratios = []
    for _ in range(4):
        screen = turbulent_screen(rng, 1_000.0, 3.0, slope)
        for sep_m in (50, 150, 300):
            k = sep_m // 10
            rms = np.sqrt(np.mean((screen[:, :-k] - screen[:, k:]) ** 2))
            ratios.append(rms / (sep_m / 1_000) ** slope)
    assert np.median(ratios) == pytest.approx(1.0, abs=0.15)


@pytest.mark.unit
def test_scans_split_the_track():
    times = np.arange(10) * 60.0
    assert scan_index(times, 0).tolist() == [0] * 10
    assert scan_index(times, 240).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]


@pytest.mark.unit
def test_antenna_errors_are_antenna_based_and_grow_with_baseline():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 400, (30, 2))
    times = np.arange(200) * 6.0
    model = CorruptionModel(
        scan_length_s=300,
        gain_amp_rms=0.1,
        gain_phase_rms_deg=5,
        slow_path_um_1km=80,
        slow_slope=0.5,
        fast_path_um_1km=40,
        fast_slope=0.5,
    )
    errors = draw_antenna_corruptions(model, xy, times, rng)
    scans = scan_index(times, 300)
    # Gains hold for a scan.
    for s in range(scans.max() + 1):
        block = errors.amplitude[:, scans == s]
        assert np.allclose(block, block[:, :1])
    assert np.std(np.log(errors.amplitude[:, 0])) == pytest.approx(0.1, abs=0.04)
    # The fast part averages to nothing over a scan; the total path difference
    # grows with baseline.
    antnum = baseline_antennas(30)
    sep = np.hypot(*(xy[antnum[:, 0]] - xy[antnum[:, 1]]).T)
    dpath = errors.path_m[antnum[:, 0]] - errors.path_m[antnum[:, 1]]
    short, long = dpath[sep < 100].std(), dpath[sep > 250].std()
    assert long > short
    gains = errors.baseline_gains(antnum, 1e-3)
    assert gains.shape == (len(antnum), len(times))
    # Closure: the phase of g12 g23 g31 is zero for antenna-based errors.
    g = {tuple(p): gains[k] for k, p in enumerate(antnum)}
    closure = np.angle(g[(0, 1)] * g[(1, 2)] * np.conj(g[(0, 2)]))
    assert np.allclose(closure, 0, atol=1e-5)


@pytest.mark.unit
def test_no_corruption_leaves_unit_gains():
    errors = draw_antenna_corruptions(
        CorruptionModel(), np.zeros((4, 2)), np.arange(5.0), np.random.default_rng(0)
    )
    assert isinstance(errors, AntennaCorruptions)
    gains = errors.baseline_gains(baseline_antennas(4), 1e-3)
    assert np.allclose(gains, 1)


def _config(**extra):
    npix = 64
    return ChannelImagingConfig(
        Npix=npix,
        Nant=4,
        Hcov=[-0.5, 0.5],
        nH=6,
        antPos=[[0.0, 0.0], [0.05, 0.0], [0.0, 0.08], [0.1, 0.1]],
        robfac=0.0,
        trlat=[np.sin(np.radians(-23.0)), np.cos(np.radians(-23.0))],
        trdec=[np.sin(np.radians(-30.0)), np.cos(np.radians(-30.0))],
        Diameters=[12.0, 0.0],
        imsize=20.0,
        Xmax=0.2,
        lfac=1.0e6,
        distmat=np.zeros((npix, npix)),
        Nphf=npix // 2,
        Np4=npix // 4,
        zooming=0,
        header={"CDELT1": -1.0 / 3600, "CDELT2": 1.0 / 3600, "NAXIS1": npix, "NAXIS2": npix},
        robust=0.0,
        scan_time_s=60.0,
        noise_entropy=(5, 1, 0),
        **extra,
    )


@pytest.mark.unit
def test_channel_applies_the_drawn_gains_to_its_rows():
    """The rows of a corrupted channel are the model times the antenna gains."""
    model = CorruptionModel(gain_amp_rms=0.1, gain_phase_rms_deg=20)
    xy_m = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 80.0], [100.0, 100.0]])
    errors = draw_antenna_corruptions(model, xy_m, np.arange(6) * 60.0, np.random.default_rng(2))
    sky = np.zeros((64, 64))
    sky[32, 32] = 1.0
    wavelength = [1.0e-6, 1.0e-6, 1.0e-6]  # km, as the simulator's
    plain = image_configured_channel(_config(), sky, wavelength, 0.0, channel=0)
    corrupted = image_configured_channel(
        _config(corruptions=model, antenna_corruptions=errors), sky, wavelength, 0.0, channel=0
    )
    rows, rows_c = plain[-1], corrupted[-1]
    expected = errors.baseline_gains(baseline_antennas(4), 1.0e-3).reshape(-1)
    valid = rows["valid"]
    assert np.allclose(rows_c["data"][valid], rows["data"][valid] * expected[valid], atol=1e-5)


@pytest.mark.unit
def test_antenna_ripple_has_its_rms_and_frequency_scale():
    """Each antenna's noise variance varies about 1 by the rms asked for, smoothly:
    its autocorrelation falls to half at the scale asked for."""
    freq = 230e9 + np.arange(4000) * 0.5e6  # 2 GHz in 0.5 MHz channels
    g = antenna_ripple(np.random.default_rng(1), 200, freq, 0.02, 100.0)
    assert g.shape == (200, 4000)
    assert abs(np.mean(g) - 1) < 0.002
    assert np.std(g) == pytest.approx(0.02, rel=0.1)
    x = g - g.mean(axis=1, keepdims=True)
    lags = np.arange(1, 600)
    ac = np.array([np.mean(x[:, :-k] * x[:, k:]) for k in lags]) / np.mean(x * x)
    half_mhz = lags[np.argmax(ac < 0.5)] * 0.5
    assert half_mhz == pytest.approx(100.0, rel=0.25)
    assert np.all(antenna_ripple(np.random.default_rng(1), 3, freq, 0.0, 100.0) == 1)


@pytest.mark.unit
def test_channel_scales_each_baselines_noise_by_its_antennas():
    """With antenna_noise, a baseline's noise is the same draw times sqrt(g_i g_j)."""
    sky = np.zeros((64, 64))
    wavelength = [1.0e-6, 1.0e-6, 1.0e-6]
    g = np.array([[1.0], [1.21], [0.81], [1.0]])  # (nant, nchan)
    plain = image_configured_channel(_config(), sky, wavelength, 1.0, channel=0)[-1]
    scaled = image_configured_channel(_config(antenna_noise=g), sky, wavelength, 1.0, channel=0)[-1]
    pairs = baseline_antennas(4)
    factor = np.repeat(np.sqrt(g[pairs[:, 0], 0] * g[pairs[:, 1], 0]), 6)
    valid = plain["valid"]
    assert np.allclose(scaled["data"][valid], plain["data"][valid] * factor[valid], atol=1e-5)
    assert not np.allclose(factor, 1)


@pytest.mark.unit
def test_simulation_parameters_take_a_preset_json_or_file(tmp_path):
    from almasim.services.interferometry.corruptions import PRESETS, resolve_corruptions

    assert resolve_corruptions("alma") == PRESETS["alma"]
    assert resolve_corruptions("none") is None
    assert resolve_corruptions('{"gain_amp_rms": 0.05}') == {"gain_amp_rms": 0.05}
    path = tmp_path / "model.json"
    path.write_text('{"slow_path_um_1km": 60}')
    assert resolve_corruptions(str(path)) == {"slow_path_um_1km": 60}
    with pytest.raises(ValueError, match="not a preset"):
        resolve_corruptions("stormy")
    assert CorruptionModel.from_mapping(PRESETS["alma"]).correlates_channels
