"""Visibility corruptions besides white thermal noise (MADIV-28)."""

from dataclasses import replace

import numpy as np
import pytest

from almasim.services.instrument.channels import averaged_correlation
from almasim.services.interferometry.corruptions import (
    AntennaCorruptions,
    CorruptionModel,
    antenna_ripple,
    baseline_antennas,
    correlated_noise,
    draw_antenna_corruptions,
    draw_bandpass_ripple,
    frozen_flow_path,
    frozen_flow_samples,
    integration_time,
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


def _correlations(kernel: np.ndarray) -> tuple[float, float, float]:
    """The kernel's output: rho1, rho2 and its variance."""
    c0, c1, c2 = kernel
    return c0 * c1 + c1 * c2, c0 * c2, float(np.sum(kernel**2))


@pytest.mark.unit
@pytest.mark.parametrize("rho", [(0.3, 0.2), (0.5, 0.4), (0.2, -0.1), (0.5, 0.0)])
def test_a_kernel_has_both_correlations_when_no_symmetric_one_does(rho):
    assert _correlations(spectral_kernel(*rho)) == pytest.approx((*rho, 1.0), abs=1e-9)


@pytest.mark.unit
def test_no_kernel_has_correlations_whose_spectrum_goes_negative():
    with pytest.raises(ValueError):
        spectral_kernel(0.9, 0.2)
    with pytest.raises(ValueError):
        spectral_kernel(0.6, 0.0)


@pytest.mark.unit
def test_the_released_models_correlations_keep_their_symmetric_kernel():
    kernel = spectral_kernel(0.6667848474717812, 0.16637600345165515)
    assert kernel[0] == kernel[2] == pytest.approx(np.sqrt(0.16637600345165515))


@pytest.mark.unit
@pytest.mark.parametrize("averaged", [2, 4, 8, 16, 32])
def test_averaged_channels_keep_their_correlations(averaged):
    rho1, rho2 = averaged_correlation(averaged)
    assert rho2 == 0.0
    kernel = spectral_kernel(rho1, rho2)
    assert _correlations(kernel) == pytest.approx((rho1, 0.0, 1.0), abs=1e-9)
    # And so does the noise drawn through it.
    shape = (400, 100)
    cube = np.stack([correlated_noise((5, averaged), c, shape, 1.0, kernel) for c in range(4)])
    lag2 = np.mean(
        [np.corrcoef(cube[c].real.ravel(), cube[c + 2].real.ravel())[0, 1] for c in (0, 1)]
    )
    assert abs(lag2) < 0.01


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
def test_baseline_errors_do_not_close_and_hold_for_a_scan():
    """Antenna gains cancel in a closure phase; the baselines' own errors do not.
    They hold for a scan and have the rms asked for."""
    xy = np.random.default_rng(0).uniform(0, 500, (12, 2))
    times = np.arange(20) * 30.0
    antennas = draw_antenna_corruptions(
        CorruptionModel(gain_phase_rms_deg=30, scan_length_s=300),
        xy,
        times,
        np.random.default_rng(3),
    )
    both = draw_antenna_corruptions(
        CorruptionModel(
            gain_phase_rms_deg=30,
            scan_length_s=300,
            baseline_amp_rms=0.05,
            baseline_phase_rms_deg=5,
        ),
        xy,
        times,
        np.random.default_rng(3),
    )
    pairs = baseline_antennas(12)
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(pairs)}

    def closure(gains):
        return np.angle(gains[index[0, 1]] * gains[index[1, 2]] * np.conj(gains[index[0, 2]]))

    g_ant, g_both = antennas.baseline_gains(pairs, 1e-3), both.baseline_gains(pairs, 1e-3)
    assert np.allclose(closure(g_ant), 0, atol=1e-5)
    assert np.std(closure(g_both)) > np.radians(3)
    own = g_both / g_ant  # the antenna draws come first, so they are the same
    assert np.allclose(own[:, :10], own[:, :1]) and np.allclose(own[:, 10:], own[:, 10:11])
    assert np.std(np.log(np.abs(own[:, 0]))) == pytest.approx(0.05, rel=0.3)
    assert np.degrees(np.std(np.angle(own[:, 0]))) == pytest.approx(5, rel=0.3)


@pytest.mark.unit
def test_baseline_errors_act_on_the_continuum_share_only():
    """With baseline_weight w, a baseline's own error b acts as 1 + w (b - 1):
    w = 0 leaves the antenna gains alone, w = 1 applies b whole."""
    xy = np.random.default_rng(0).uniform(0, 500, (6, 2))
    model = CorruptionModel(gain_phase_rms_deg=10, baseline_amp_rms=0.2, baseline_phase_rms_deg=20)
    errors = draw_antenna_corruptions(model, xy, np.arange(4) * 30.0, np.random.default_rng(1))
    pairs = baseline_antennas(6)
    whole = errors.baseline_gains(pairs, 1e-3)
    none = errors.baseline_gains(pairs, 1e-3, baseline_weight=0.0)
    half = errors.baseline_gains(pairs, 1e-3, baseline_weight=0.5)
    g = errors.amplitude * np.exp(1j * (errors.phase + 2 * np.pi * errors.path_m / 1e-3))
    assert np.allclose(none, g[pairs[:, 0]] * np.conj(g[pairs[:, 1]]), atol=1e-5)
    assert np.allclose(half, (none + whole) / 2, atol=1e-5)
    assert not np.allclose(whole, none)


@pytest.mark.unit
def test_interferometer_hands_its_channels_the_corruption_fields():
    """The fields every channel task gets: the continuum fraction when it covers
    the cube's channels, the antenna errors and each antenna's noise ripple."""
    from types import SimpleNamespace

    from almasim.services.interferometry.core import Interferometer

    freqs = 230e9 + np.arange(8) * 1e6
    sim = SimpleNamespace(
        corruptions=CorruptionModel(baseline_amp_rms=0.05, antenna_ripple_rms=0.02),
        noise_entropy=(7, 1),
        continuum_fraction=np.linspace(1.0, 0.5, 8),
        Nchan=8,
        nH=4,
        scan_time=30.0,
        sample_offsets_s=None,
        antPos=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        Nant=3,
        antenna_diameter_m=12.0,
        antenna_diameters_m=np.full(3, 12.0),
        spectral_window=SimpleNamespace(frequencies_hz=lambda: freqs),
    )
    fields = Interferometer._corruption_fields(sim)
    assert np.allclose(fields["continuum_fraction"], np.linspace(1.0, 0.5, 8))
    assert fields["antenna_corruptions"].baseline is not None
    assert fields["antenna_noise"].shape == (3, 8)
    sim.continuum_fraction = np.ones(5)  # not this cube's channels: left out
    assert "continuum_fraction" not in Interferometer._corruption_fields(sim)
    sim.corruptions = None
    assert Interferometer._corruption_fields(sim) == {}


@pytest.mark.unit
def test_continuum_fraction_follows_the_lines():
    """The continuum's share of each channel: 1 away from the lines, lower on them."""
    from almasim.services.simulation.spectral_setup import SpectralSetup

    spectral = SpectralSetup(
        continuum=np.full(40, 0.01),
        line_fluxes=[1.0],
        line_names=["CO"],
        line_frequency=None,
        redshift=0.0,
        rest_frequency=None,
        source_channel_index=[20],
        fwhm_z=[4.0],
        lum_infrared=None,
    )
    fraction = spectral.continuum_fraction(40)
    assert fraction.shape == (40,)
    assert np.allclose(fraction[:5], 1.0, atol=1e-6) and np.allclose(fraction[-5:], 1.0, atol=1e-6)
    assert fraction[20] < 0.1
    assert spectral.continuum_fraction(39) is None


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
    # Its rows' sigma carries the same factor: the MS writer's hands draw at it.
    np.testing.assert_allclose(scaled["sigma"], plain["sigma"] * factor, rtol=1e-6)


@pytest.mark.unit
def test_simulation_parameters_take_a_model_json_or_file(tmp_path):
    from almasim.services.interferometry.corruptions import resolve_corruptions

    # alma@1 sets the preset's values; "alma" names the newest version.
    assert CorruptionModel.from_mapping(resolve_corruptions("alma@1")) == CorruptionModel(
        spectral_rho1=2 / 3,
        spectral_rho2=1 / 6,
        scan_length_s=300.0,
        gain_amp_rms=0.08,
        slow_path_um_1km=80.0,
        slow_slope=0.45,
        fast_path_um_1km=40.0,
        fast_slope=0.3,
        fast_coherence_s=10.0,
        atmospheric_tsys=True,
        receiver_ripple_rms=0.02,
        antenna_ripple_rms=0.02,
        antenna_ripple_mhz=120.0,
        baseline_amp_rms=0.06,
        baseline_phase_rms_deg=2.5,
        radiometer_noise=True,
    )
    assert resolve_corruptions("none") is None
    as_model = CorruptionModel.from_mapping
    assert as_model(resolve_corruptions('{"gain_amp_rms": 0.05}')) == CorruptionModel(
        gain_amp_rms=0.05
    )
    path = tmp_path / "model.json"
    path.write_text('{"slow_path_um_1km": 60}')
    assert as_model(resolve_corruptions(str(path))) == CorruptionModel(slow_path_um_1km=60.0)
    with pytest.raises(ValueError, match="not a released model"):
        resolve_corruptions("stormy")


@pytest.mark.unit
def test_correlation_follows_the_windows_averaging_unless_set():
    from almasim.services.instrument.channels import averaged_correlation

    model = CorruptionModel(spectral_rho1=2 / 3, spectral_rho2=1 / 6)
    assert model.channel_correlation() == (2 / 3, 1 / 6)
    assert model.for_window(4).channel_correlation() == pytest.approx(averaged_correlation(4))
    fixed = CorruptionModel.from_mapping(
        {"spectral_rho1": 2 / 3, "spectral_rho2": 1 / 6, "spectral_average": "2"}
    )
    assert fixed.spectral_average == 2
    assert fixed.for_window(16).channel_correlation() == pytest.approx(averaged_correlation(2))


@pytest.mark.unit
def test_each_antenna_error_has_its_own_stream():
    """Switching one term off, or changing its strength, leaves the others'
    draws as they were: the terms are separable."""
    xy = np.random.default_rng(0).uniform(0, 800, (8, 2))
    times = np.arange(30) * 20.0
    full = CorruptionModel(
        scan_length_s=200,
        gain_amp_rms=0.08,
        gain_phase_rms_deg=10,
        slow_path_um_1km=80,
        fast_path_um_1km=40,
        baseline_amp_rms=0.06,
        baseline_phase_rms_deg=2.5,
    )

    def draw(**changes):
        return draw_antenna_corruptions(
            replace(full, **changes), xy, times, np.random.default_rng(5)
        )

    everything = draw()
    no_gains = draw(gain_amp_rms=0.0, gain_phase_rms_deg=0.0)
    assert np.array_equal(no_gains.path_m, everything.path_m)
    assert np.array_equal(no_gains.baseline, everything.baseline)
    assert np.all(no_gains.amplitude == 1) and not np.array_equal(
        no_gains.amplitude, everything.amplitude
    )
    no_slow = draw(slow_path_um_1km=0.0)
    assert np.array_equal(no_slow.amplitude, everything.amplitude)
    assert np.array_equal(no_slow.baseline, everything.baseline)
    only_fast = draw(slow_path_um_1km=0.0, gain_amp_rms=0.0, gain_phase_rms_deg=0.0)
    assert np.allclose(only_fast.path_m, no_slow.path_m)
    stronger_fast = draw(fast_path_um_1km=80.0)
    assert np.array_equal(stronger_fast.phase, everything.phase)
    assert np.array_equal(stronger_fast.baseline, everything.baseline)


@pytest.mark.unit
def test_an_observation_takes_its_dishs_in_scan_strength_and_its_own_draw():
    """Each dish size has its own in-scan strength, and each observation draws
    its own from the spread; without a spread the strength is the dish's."""
    model = CorruptionModel.from_mapping(
        {"fast_path_um_1km": 100.0, "fast_path_by_dish": [[12.0, 65.0], [7.0, 150.0]]}
    )
    assert model.fast_path_by_dish == ((12.0, 65.0), (7.0, 150.0))
    rng = np.random.default_rng(0)
    assert model.for_observation(12.0, rng).fast_path_um_1km == 65.0
    assert model.for_observation(7.0, rng).fast_path_um_1km == 150.0
    assert model.for_observation(10.0, rng).fast_path_um_1km == 100.0  # no row: the default
    spread = replace(model, fast_log_spread=0.7)
    draws = [spread.for_observation(12.0, np.random.default_rng(k)) for k in range(400)]
    paths = np.array([d.fast_path_um_1km for d in draws])
    assert np.std(np.log(paths)) == pytest.approx(0.7, rel=0.1)
    assert np.exp(np.median(np.log(paths))) == pytest.approx(65.0, rel=0.1)
    # The same seed, the same observation; and the result draws nothing more.
    again = spread.for_observation(12.0, np.random.default_rng(3))
    assert again == draws[3] and again.fast_log_spread == 0.0 and not again.fast_path_by_dish
    assert again.for_observation(7.0, np.random.default_rng(9)) == again
    # A {diameter: strength} object reads the same.
    assert CorruptionModel.from_mapping(
        {"fast_path_by_dish": {"12": 65.0, "7": 150.0}}
    ).fast_path_by_dish == ((12.0, 65.0), (7.0, 150.0))


@pytest.mark.unit
def test_the_interferometer_draws_its_observations_in_scan_strength():
    """The channels get the observation's own strength; the other antenna
    errors' draws do not depend on it."""
    from types import SimpleNamespace

    from almasim.services.interferometry.core import Interferometer

    base = CorruptionModel(fast_path_um_1km=80.0, fast_slope=0.5, gain_amp_rms=0.05)
    sim = SimpleNamespace(
        corruptions=replace(base, fast_path_by_dish=((7.0, 160.0),), fast_log_spread=0.5),
        noise_entropy=(7, 1),
        continuum_fraction=None,
        Nchan=4,
        nH=6,
        scan_time=6.0,
        sample_offsets_s=None,
        antPos=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.3, 0.1, 0.0]],
        Nant=4,
        antenna_diameter_m=7.0,
        antenna_diameters_m=np.full(4, 7.0),
        spectral_window=SimpleNamespace(frequencies_hz=lambda: np.zeros(4)),
    )
    fields = Interferometer._corruption_fields(sim)
    drawn = fields["corruptions"].fast_path_um_1km
    # The observation's own stream: (entropy, 1_000_000, 4).
    z = np.random.default_rng((7, 1, 1_000_000, 4)).standard_normal()
    assert drawn == pytest.approx(160.0 * np.exp(0.5 * z))
    assert fields["corruptions"] == Interferometer._corruption_fields(sim)["corruptions"]
    sim.corruptions = replace(base, fast_path_um_1km=drawn)
    same = Interferometer._corruption_fields(sim)["antenna_corruptions"]
    assert np.allclose(fields["antenna_corruptions"].amplitude, same.amplitude)
    assert np.allclose(fields["antenna_corruptions"].path_m, same.path_m)


@pytest.mark.unit
def test_instrumental_non_closing_errors_act_on_lines_and_continuum_alike():
    """Closure departures as configured: the continuum term moves only the
    continuum share's closures (none for a line-only channel), the instrumental
    term moves a line-only and a continuum-only channel's alike, and adding it
    leaves the continuum term's own draws as they were."""
    xy = np.random.default_rng(0).uniform(0, 500, (10, 2))
    times = np.arange(20) * 30.0
    pairs = baseline_antennas(10)
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(pairs)}

    def closure(gains):
        return np.angle(gains[index[0, 1]] * gains[index[1, 2]] * np.conj(gains[index[0, 2]]))

    base = dict(gain_phase_rms_deg=30, scan_length_s=60)
    continuum = CorruptionModel(**base, baseline_amp_rms=0.05, baseline_phase_rms_deg=5)
    instrumental = CorruptionModel(
        **base, baseline_instrumental_amp_rms=0.03, baseline_instrumental_phase_rms_deg=4
    )
    both = replace(continuum, baseline_instrumental_amp_rms=0.03,
                   baseline_instrumental_phase_rms_deg=4)  # fmt: skip
    drawn = {
        name: draw_antenna_corruptions(m, xy, times, np.random.default_rng(3))
        for name, m in (("continuum", continuum), ("instrumental", instrumental), ("both", both))
    }
    for name, errors in drawn.items():
        line = closure(errors.baseline_gains(pairs, 1e-3, baseline_weight=0.0))
        cont = closure(errors.baseline_gains(pairs, 1e-3, baseline_weight=1.0))
        if name == "continuum":
            assert np.allclose(line, 0, atol=1e-5)  # a line-only source closes
            assert np.std(cont) > np.radians(3)
        else:
            assert np.std(line) > np.radians(2) and np.std(cont) > np.radians(2)
    assert drawn["continuum"].instrumental is None and drawn["instrumental"].baseline is None
    assert np.array_equal(drawn["both"].baseline, drawn["continuum"].baseline)
    assert np.array_equal(drawn["both"].instrumental, drawn["instrumental"].instrumental)
    own = drawn["instrumental"].instrumental[pairs[:, 0], pairs[:, 1], 0]
    assert np.std(np.log(np.abs(own))) == pytest.approx(0.03, rel=0.3)
    assert np.degrees(np.std(np.angle(own))) == pytest.approx(4, rel=0.3)


def _frozen_model(**overrides):
    values = dict(fast_path_um_1km=300.0, fast_slope=0.5, fast_wind_mps=10.0, outer_scale_km=3.0)
    return CorruptionModel(**{**values, **overrides})


@pytest.mark.unit
def test_frozen_flow_coherence_follows_the_gaussian_law():
    """Each integration's averaged visibility, computed sample by sample over the
    screen that passes, loses exp(-sigma^2 / 2) of its amplitude, sigma the
    phase rms within it: what baseline_gains applies from path_var_m2."""
    rng = np.random.default_rng(3)
    xy = rng.uniform(0.0, 1_500.0, (20, 2))
    times = np.arange(40) * 6.0
    scans = scan_index(times, 120.0)
    model = _frozen_model()
    samples = frozen_flow_samples(model, xy, times, scans, np.random.default_rng(5))
    assert samples.shape[0] == 24  # 4 per 10 m pixel: 60 m per 6 s integration
    _, var = frozen_flow_path(model, xy, samples, scans)
    i, j = np.triu_indices(len(xy), 1)
    raw = samples[:, :, i] - samples[:, :, j]  # (nsub, ntime, nbl)
    raw_var = raw.var(axis=0)
    ok = raw_var > 0
    # The returned variance is the raw one scaled as the path is.
    factor2 = np.median(var[i, j, :].T[ok] / raw_var[ok])
    assert np.allclose(var[i, j, :].T[ok], factor2 * raw_var[ok], rtol=1e-4)
    k = 2 * np.pi / 0.87e-3  # band 7
    explicit = np.abs(np.mean(np.exp(1j * k * raw * np.sqrt(factor2)), axis=0))
    predicted = np.exp(-0.5 * k**2 * var[i, j, :].T)
    assert 0.3 < np.median(predicted) < 0.99  # the decorrelation is visible
    assert np.mean(explicit) == pytest.approx(np.mean(predicted), rel=0.03)


@pytest.mark.unit
def test_frozen_flow_temporal_structure_follows_the_screen_along_the_wind():
    """An antenna's path difference tau apart is the screen's v tau apart:
    path_um_1km (v tau / 1 km)^slope, as the array's spatial law."""
    rng = np.random.default_rng(11)
    xy = rng.uniform(0.0, 1_000.0, (24, 2))
    times = np.arange(600) * 1.0
    scans = np.zeros(times.size, dtype=int)
    model = _frozen_model()
    samples = frozen_flow_samples(model, xy, times, scans, np.random.default_rng(2))
    path, _ = frozen_flow_path(model, xy, samples, scans)
    for lag in (10, 20, 30):  # 100 to 300 m along the wind
        rms = np.sqrt(np.mean((path[:, lag:] - path[:, :-lag]) ** 2))
        expected = model.fast_path_um_1km * 1e-6 * (10.0 * lag / 1_000.0) ** model.fast_slope
        assert rms == pytest.approx(expected, rel=0.3)


@pytest.mark.unit
def test_frozen_flow_replaces_the_blended_screen_and_reaches_the_gains():
    xy = np.random.default_rng(0).uniform(0.0, 800.0, (8, 2))
    times = np.arange(30) * 6.0
    blended = draw_antenna_corruptions(
        _frozen_model(fast_wind_mps=0.0), xy, times, np.random.default_rng(1)
    )
    assert blended.path_var_m2 is None
    frozen = draw_antenna_corruptions(_frozen_model(), xy, times, np.random.default_rng(1))
    assert frozen.path_var_m2.shape == (8, 8, 30)
    assert np.allclose(np.diagonal(frozen.path_var_m2), 0.0, atol=1e-18)
    assert integration_time(times, np.zeros(30, dtype=int)) == 6.0
    pairs = baseline_antennas(8)
    loose = replace(frozen, path_var_m2=None).baseline_gains(pairs, 0.87e-3)
    gains = frozen.baseline_gains(pairs, 0.87e-3)
    # The within-integration loss only lowers amplitudes, never raises them.
    assert np.all(np.abs(gains) <= np.abs(loose) + 1e-6)
    assert np.mean(np.abs(gains)) < np.mean(np.abs(loose))


BAND1_RIPPLE = ((12.0, 50.0, 25.0, 0.01, 3.0), (7.0, 50.0, 40.0, 0.01, 3.0))


@pytest.mark.unit
def test_the_bandpass_ripple_has_its_dishs_period_and_peaks():
    """Recovered across a band 1 window: each dish size's period (25 MHz on
    12 m, 40 MHz on 7 m), peak amplitude 1% and phase 3 degrees; none above
    the edge."""
    dishes = np.array([12.0] * 6 + [7.0] * 4)
    ripple = draw_bandpass_ripple(BAND1_RIPPLE, dishes, np.random.default_rng(4))
    freq = 40e9 + np.arange(4096) * 0.25e6  # 1 GHz in 0.25 MHz channels
    b = ripple.factors(freq)
    assert np.max(np.abs(np.abs(b) - 1.0)) == pytest.approx(0.01, rel=0.01)
    assert np.degrees(np.max(np.abs(np.angle(b)))) == pytest.approx(3.0, rel=0.01)
    spectrum = np.abs(np.fft.rfft(np.abs(b) - 1.0, axis=1))
    period_mhz = 1.0 / np.fft.rfftfreq(freq.size, 0.25)[np.argmax(spectrum[:, 1:], axis=1) + 1]
    assert np.allclose(period_mhz[:6], 25.0, rtol=0.03)
    assert np.allclose(period_mhz[6:], 40.0, rtol=0.03)
    assert np.allclose(ripple.factors(np.array([90e9, 230e9])), 1.0)  # bands 3 and 6
    alone = draw_bandpass_ripple(BAND1_RIPPLE[:1], dishes, np.random.default_rng(4))
    assert np.allclose(alone.factors(freq)[6:], 1.0)  # a dish without a row


@pytest.mark.unit
def test_the_bandpass_ripple_reaches_the_baselines_from_a_stream_of_its_own():
    xy = np.random.default_rng(0).uniform(0, 500, (6, 2))
    times = np.arange(10) * 30.0
    model = CorruptionModel(gain_amp_rms=0.05, scan_length_s=60)
    rippled = replace(model, bandpass_ripple=BAND1_RIPPLE)
    assert rippled.has_antenna_errors
    plain = draw_antenna_corruptions(model, xy, times, np.random.default_rng(2))
    errors = draw_antenna_corruptions(rippled, xy, times, np.random.default_rng(2))
    assert plain.bandpass is None
    assert np.array_equal(plain.amplitude, errors.amplitude)  # the gains' draws unchanged
    pairs = baseline_antennas(6)
    wavelength = 299_792_458.0 / 42e9
    b = errors.bandpass.factors(np.array([42e9]))[:, 0]
    expected = (
        plain.baseline_gains(pairs, wavelength)
        * (b[pairs[:, 0]] * np.conj(b[pairs[:, 1]]))[:, None]
    )
    assert np.allclose(errors.baseline_gains(pairs, wavelength), expected, atol=1e-6)
    band3 = 299_792_458.0 / 100e9
    assert np.allclose(errors.baseline_gains(pairs, band3), plain.baseline_gains(pairs, band3))
    sevens = draw_antenna_corruptions(
        rippled, xy, times, np.random.default_rng(2), diameters_m=np.full(6, 7.0)
    )
    assert np.all(sevens.bandpass.period_hz == 40e6)
