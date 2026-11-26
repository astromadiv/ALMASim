"""Reducing windows' measurements into a candidate noise model (MADIV-28)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from almasim.services.interferometry.atmosphere import system_temperature
from almasim.services.noise.aggregate import Measured, aggregate
from almasim.services.noise.measurements import (
    AntennaNoise,
    AntennaShape,
    BandpassFit,
    Exposure,
    FitBin,
    LineNonClosing,
    PhaseFit,
    ThermalNoise,
    Tsys,
    VisibilityNoise,
    WindowFit,
    WindowInput,
    WindowNoise,
)
from almasim.services.noise_model import released_model, resolve_noise_model

C = 299_792_458.0
NCHAN = 256


def _window(
    k: int,
    *,
    ghz: float = 230.0,
    dish: float = 12.0,
    trx: float = 40.0,
    ripple: float = 0.02,
    forward_efficiency: float | None = None,
    efficiency_scale: float = 1.0,
    tdm: float | None = None,
) -> WindowNoise:
    rng = np.random.default_rng(k)
    x = np.linspace(-1, 1, NCHAN)
    tsys_shape = 1 + 0.05 * x  # Tsys^2 slope, as SYSCAL shows it
    receiver = 1 + ripple * np.sqrt(2) * np.cos(np.pi * (x + rng.uniform(0, 1)))  # rms ~ ripple
    profile = tsys_shape * receiver**2 + rng.normal(0, 0.002, NCHAN)
    # TDM: native 15.625 MHz channels (resolution twice the width), ``tdm``
    # times the efficiency of FDM's.
    width = 7.8125e6 if tdm is None else 15.625e6
    freq = ghz * 1e9 + np.linspace(-0.5, 0.5, 64) * NCHAN * width
    elevation, pwv = 60.0, 1.2
    airmass = 1 / np.sin(np.radians(elevation))
    tsys = float(
        np.median(
            system_temperature(
                freq, trx, pwv_mm=pwv, airmass=airmass, forward_efficiency=forward_efficiency
            )
        )
    )
    eta0, eps = (0.75, 26e-6) if dish == 12 else (0.7, 24e-6)
    aperture = eta0 * np.exp(-((4 * np.pi * eps * ghz * 1e9 / C) ** 2)) * rng.normal(1, 0.02)
    aperture *= efficiency_scale  # the band's own departure from the dish's curve
    aperture *= tdm or 1.0
    return WindowNoise(
        name=f"w{k}",
        almasim_version="test",
        input=WindowInput(ms=f"w{k}.ms", spw=1, pwv_mm=pwv),
        exposure=Exposure(
            antennas=40,
            dish_diameter_m=dish,
            baselines=780,
            integrations=500,
            integration_s=6.0,
            time_on_source_s=3000.0,
            channels=NCHAN,
            channel_width_hz=width,
            noise_bandwidth_hz=width,
            resolution_hz=width if tdm is None else 2 * width,
            frequency_ghz=ghz,
            correlations=2,
            flagged_fraction=0.01,
            elevation_deg=elevation,
        ),  # fmt: skip
        thermal=ThermalNoise(
            scale=1.0,
            rho1=0.667 + rng.normal(0, 0.005),
            rho2=0.166 + rng.normal(0, 0.005),
            channel_profile=list(profile),
            rows_sampled=10**6,
            row_step=1,
        ),  # fmt: skip
        visibility=VisibilityNoise(
            sigma_jy=1.0, unit_jy=1.0, hands_identical=False, efficiency=aperture * 0.96 * 0.88
        ),
        tsys=Tsys(median_k=tsys, profile=list(tsys_shape / tsys_shape.mean()), spectral_window=3),
        antenna_shape=AntennaShape(
            antennas_used=40,
            channels_used=NCHAN - 3,
            excess_rms=0.03 + 0.001 * k,
            half_width_channels=10,
            half_width_mhz=100.0 + k,
        ),  # fmt: skip
        antenna_noise=AntennaNoise(
            antennas_used=40, baselines_used=780, sefd_log_rms=0.1 + 0.001 * k
        ),
    )


def _fit(k: int) -> WindowFit:
    return WindowFit(
        name=f"w{k}",
        bright_groups=500,
        continuum_snr=400.0,
        realizations=2,
        kappa=2.67,
        bins=[
            FitBin(
                baseline_m=100.0,
                groups=50,
                phase_rms_deg=5.0,
                thermal_phase_rms_deg=1.0,
                lnamp_mean=-0.01,
                thermal_lnamp_mean=0.0,
            )
        ],
        slow_phase=PhaseFit(
            bins=6, slope=0.4 + 0.01 * k, phase_1km_deg=20.0, path_um_1km=100.0 + 10 * k
        ),
        gain_amp_rms=0.05 + 0.01 * k,
        baseline_amp_rms=0.04,
        baseline_phase_rms_deg=3.0,
    )


@pytest.mark.unit
def test_the_candidate_takes_the_windows_medians():
    windows = [Measured(_window(k), _fit(k) if k < 4 else None) for k in range(8)]
    windows += [Measured(_window(10 + k, dish=7.0, ghz=100.0 + 60 * k)) for k in range(6)]
    windows += [Measured(_window(20 + k, ghz=100.0 + 60 * k)) for k in range(6)]
    base = released_model("alma", 1)
    candidate = aggregate(windows, base, data="/runs/x")
    assert (candidate.name, candidate.version) == ("alma-candidate", 2)
    c = candidate.components
    assert c.channel_correlation.rho1 == pytest.approx(0.667, abs=0.01)
    assert c.receiver_ripple.rms == pytest.approx(0.02, rel=0.35)
    assert c.antenna_ripple.rms == pytest.approx(
        np.median(
            [0.03 + 0.001 * k for k in range(8)]
            + [0.03 + 0.001 * k for k in list(range(10, 16)) + list(range(20, 26))]
        )
    )
    # antenna_sefd: alma@1 has none, so the candidate adds it with the median.
    assert base.components.antenna_sefd is None
    ks = list(range(8)) + list(range(10, 16)) + list(range(20, 26))
    assert c.antenna_sefd.log_rms == pytest.approx(np.median([0.1 + 0.001 * k for k in ks]))
    assert c.slow_phase.path_um_1km == pytest.approx(115.0)
    assert c.antenna_gains.amp_rms == pytest.approx(0.065)
    assert c.baseline_errors.phase_rms_deg == pytest.approx(3.0)
    # The receiver temperature the Tsys implies, in the 12 m windows' band (6).
    trx = {r.bands: r.trx_k for r in c.radiometer.receiver_temperature_k}
    assert trx["6"] == pytest.approx(40.0, rel=0.05)
    assert trx["9, 10"] == 160.0  # no windows there: the base's
    aperture = {a.diameter_m: a for a in c.radiometer.aperture}
    assert aperture[12.0].eta0 == pytest.approx(0.75, abs=0.03)
    assert aperture[12.0].surface_rms_m == pytest.approx(26e-6, rel=0.2)
    assert aperture[7.0].eta0 == pytest.approx(0.7, abs=0.04)
    values = candidate.provenance.values
    assert values["slow_phase.path_um_1km"].n == 4 and values["channel_correlation.rho1"].n == 20
    assert candidate.provenance.windows == 20 and candidate.provenance.data == "/runs/x"
    assert "fast_phase.path_um_1km: 0 window(s)" in candidate.provenance.notes
    assert c.fast_phase == base.components.fast_phase
    # The candidate is a model like any other: it validates and simulates.
    assert resolve_noise_model(candidate.model_dump_json()).engine_fields()[
        "gain_amp_rms"
    ] == pytest.approx(0.065)


def _bandpass(period_mhz: float, amp_rms: float, phase_rms_deg: float = 2.0) -> BandpassFit:
    return BandpassFit(
        chunks=240,
        chunk_mhz=3.9,
        antennas=40,
        baselines=780,
        amp_rms=amp_rms,
        phase_rms_deg=phase_rms_deg,
        thermal_amp_rms=0.01,
        thermal_phase_rms_deg=0.5,
        period_mhz=period_mhz,
    )


@pytest.mark.unit
def test_the_fits_set_the_wind_the_instrument_s_share_and_a_seen_ripple():
    base = released_model("alma", 1)
    winds = [2.0, 7.0, 7.0, 10.0, 15.0]
    shares = [0.8, 0.9, 1.3, 0.95, None]  # 1.3: noise in the split, counts as 1
    continuum = [0.1, 0.3, 0.5, 0.2, 0.0]
    measured = []
    for k, (wind, share) in enumerate(zip(winds, shares, strict=True)):
        fit = _fit(k).model_copy(
            update={
                "fast_phase": PhaseFit(
                    bins=40, slope=0.5, phase_1km_deg=5.0, path_um_1km=50.0, wind_mps=wind
                ),
                "line_nonclosing": LineNonClosing(
                    groups=100,
                    continuum_share=continuum[k],
                    amp_rms=0.04,
                    phase_rms_deg=3.0,
                    instrumental_amp_rms=None if share is None else 0.04 * share,
                    instrumental_phase_rms_deg=None if share is None else 1.5,
                ),
                # 12 m: three windows see a 10 MHz ripple, two a smooth shape.
                "bandpass": _bandpass(10.0 if k < 3 else 400.0, 0.01 * (k + 1)),
            }
        )
        measured.append(Measured(_window(k, ghz=100.0 + 50 * k), fit))
    # A 12 m window whose ripple is below the noise's own rms does not see one.
    measured.append(
        Measured(
            _window(8, ghz=400.0),
            _fit(8).model_copy(update={"bandpass": _bandpass(10.0, 0.005, 0.3)}),
        )
    )
    # Line channels without continuum: their window's error may be the line's.
    measured.append(
        Measured(
            _window(7),
            _fit(7).model_copy(
                update={
                    "line_nonclosing": LineNonClosing(
                        groups=100,
                        continuum_share=0.01,
                        amp_rms=0.04,
                        phase_rms_deg=3.0,
                        instrumental_amp_rms=0.04,
                        instrumental_phase_rms_deg=3.0,
                    )
                }
            ),
        )
    )
    # 7 m: one window with a ripple is too few.
    measured.append(
        Measured(
            _window(9, dish=7.0),
            _fit(9).model_copy(update={"bandpass": _bandpass(20.0, 0.05)}),
        )
    )
    candidate = aggregate(measured, base)
    c = candidate.components
    assert c.fast_phase.wind_mps == 7.0
    errors, values = c.baseline_errors, candidate.provenance.values
    # Shares 0.8, 0.9, 1.0, 0.95: median 0.925 of the windows' amplitude error.
    share = values["baseline_errors.instrumental_amp_rms_share"]
    assert (share.n, share.median) == (4, pytest.approx(0.925))
    assert errors.instrumental_amp_rms == pytest.approx(0.925 * 0.04)
    assert errors.amp_rms == pytest.approx(0.04 * np.sqrt(1 - 0.925**2))
    # The continuum still sees the windows' median in quadrature.
    assert np.hypot(errors.amp_rms, errors.instrumental_amp_rms) == pytest.approx(0.04)
    assert errors.instrumental_phase_rms_deg == pytest.approx(1.5)
    assert errors.phase_rms_deg == pytest.approx(np.sqrt(3.0**2 - 1.5**2))
    (row,) = c.antenna_gains.bandpass
    assert (row.diameter_m, row.period_mhz, row.upper_ghz) == (12.0, 10.0, 200.0)
    assert row.amp == pytest.approx(np.sqrt(2) * 0.02)
    assert row.phase_deg == pytest.approx(np.sqrt(2) * 2.0)
    assert "antenna_gains.bandpass.7m: 1 window(s) see a ripple" in candidate.provenance.notes
    assert resolve_noise_model(candidate.model_dump_json()).engine_fields()[
        "baseline_instrumental_amp_rms"
    ] == pytest.approx(0.037)


@pytest.mark.unit
def test_gain_and_baseline_errors_come_from_the_bright_windows():
    base = released_model("alma", 1)
    faint = {"gain_amp_rms": 0.3, "baseline_amp_rms": 0.4, "baseline_phase_rms_deg": 20.0}
    measured = [Measured(_window(k), _fit(k)) for k in range(3)]
    # Faint windows, and one whose task has no continuum S/N, read too large.
    measured += [
        Measured(_window(k), _fit(k).model_copy(update={**faint, "continuum_snr": snr}))
        for k, snr in ((3, 249.0), (4, 30.0), (5, 8.0), (6, None))
    ]
    candidate = aggregate(measured, base)
    values = candidate.provenance.values
    assert values["antenna_gains.amp_rms"].n == 3
    assert candidate.components.antenna_gains.amp_rms == pytest.approx(0.06)
    assert values["baseline_errors.amp_rms"].median == pytest.approx(0.04)
    assert values["baseline_errors.phase_rms_deg"].median == pytest.approx(3.0)
    # The phase screens are every fitted window's.
    assert values["slow_phase.path_um_1km"].n == 7

    everyone = aggregate(measured, base, bright_snr=0.0)
    assert everyone.provenance.values["antenna_gains.amp_rms"].n == 6
    assert everyone.components.antenna_gains.amp_rms > 0.06
    # Too few bright windows keep the base's values.
    few = aggregate(measured, base, bright_snr=1000.0)
    assert few.components.antenna_gains.amp_rms == base.components.antenna_gains.amp_rms
    assert "antenna_gains.amp_rms: 0 window(s), kept the base's" in few.provenance.notes


@pytest.mark.unit
def test_the_aperture_does_not_depend_on_the_gains():
    """The residual gains scale the simulated sky, not its noise: the aperture
    fitted to the real noise is the same with or without them."""
    windows = [Measured(_window(20 + k, ghz=100.0 + 60 * k), _fit(k)) for k in range(6)]
    base = released_model("alma", 1)
    with_gains = aggregate(windows, base)
    without = aggregate(windows, base.select(without=["antenna_gains", "baseline_errors"]))
    assert with_gains.components.antenna_gains.amp_rms > 0

    def dish(model):
        return {x.diameter_m: x for x in model.components.radiometer.aperture}[12.0]

    assert dish(with_gains).eta0 == pytest.approx(dish(without).eta0, rel=1e-9)
    assert "gains" not in with_gains.provenance.notes


@pytest.mark.unit
def test_the_receiver_temperature_follows_the_bases_tsys_equation():
    """A base with a forward efficiency (the Technical Handbook's Tsys) gets the
    receiver temperatures that equation needs."""
    alma = released_model("alma", 1)
    radiometer = json.loads(alma.model_dump_json())["components"]["radiometer"]
    base = resolve_noise_model(
        {
            **json.loads(alma.model_dump_json()),
            "components": {"radiometer": {**radiometer, "forward_efficiency": 0.95}},
        }
    )
    windows = [Measured(_window(k, trx=35.0, forward_efficiency=0.95)) for k in range(4)]
    candidate = aggregate(windows, base)
    trx = {r.bands: r.trx_k for r in candidate.components.radiometer.receiver_temperature_k}
    assert trx["6"] == pytest.approx(35.0, rel=1e-6)
    assert candidate.components.radiometer.forward_efficiency == 0.95


@pytest.mark.unit
def test_a_band_whose_windows_need_more_at_higher_frequencies_gets_a_slope():
    """Enough windows across a band whose needed receiver temperature rises with
    frequency give the band a slope about their median frequency; a band whose
    windows need one temperature keeps one."""
    base = released_model("alma", 1)
    ghz = np.linspace(290.0, 360.0, 10)
    rising = [Measured(_window(k, ghz=f, trx=68.0 + 0.4 * (f - 325.0))) for k, f in enumerate(ghz)]
    level = [
        Measured(_window(20 + k, ghz=f, trx=40.0)) for k, f in enumerate(np.linspace(215, 270, 10))
    ]
    candidate = aggregate(rising + level, base)
    rows = {r.bands: r for r in candidate.components.radiometer.receiver_temperature_k}
    assert rows["7"].slope_k_per_ghz == pytest.approx(0.4, rel=0.02)
    assert rows["7"].center_ghz == pytest.approx(325.0)
    assert rows["7"].trx_k == pytest.approx(68.0, rel=0.01)
    assert rows["6"].slope_k_per_ghz is None and rows["6"].center_ghz is None
    assert rows["6"].trx_k == pytest.approx(40.0, rel=0.01)
    values = candidate.provenance.values
    slope = values["radiometer.receiver_temperature_k[7].slope_k_per_ghz"]
    assert slope.n == 10 and slope.q16 <= slope.median <= slope.q84
    assert "radiometer.receiver_temperature_k[6].slope_k_per_ghz" not in values
    # Too few windows to tell a slope: one temperature, the windows' median.
    few = aggregate(rising[:5], base)
    rows = {r.bands: r for r in few.components.radiometer.receiver_temperature_k}
    assert rows["7"].slope_k_per_ghz is None
    # A base's slope is the windows' to set again: a level band drops it.
    data = json.loads(candidate.model_dump_json())
    for r in data["components"]["radiometer"]["receiver_temperature_k"]:
        if r["bands"] == "6":
            r.update(slope_k_per_ghz=0.3, center_ghz=240.0)
    again = aggregate(level, resolve_noise_model(data))
    rows = {r.bands: r for r in again.components.radiometer.receiver_temperature_k}
    assert rows["6"].slope_k_per_ghz is None and rows["7"].slope_k_per_ghz is not None


@pytest.mark.unit
def test_the_slope_is_the_one_the_noise_needs():
    """The slope follows the windows' noise, not their Tsys alone: windows of one
    Tsys-implied temperature whose efficiency falls with frequency (their noise
    rises across the band) get a slope, and the band's temperature stays the
    one their Tsys implies."""
    base = released_model("alma", 1)
    ghz = np.linspace(290.0, 360.0, 10)
    windows = [
        Measured(_window(k, ghz=f, trx=68.0, efficiency_scale=1.0 - 0.002 * (f - 325.0)))
        for k, f in enumerate(ghz)
    ]
    # Windows of the other bands hold the dish's curve in place.
    others = [
        Measured(_window(30 + k, ghz=f))
        for k, f in enumerate(
            [100.0, 105.0, 150.0, 155.0, 230.0, 235.0, 410.0, 420.0, 650.0, 660.0]
        )
    ]
    candidate = aggregate(windows + others, base)
    rows = {r.bands: r for r in candidate.components.radiometer.receiver_temperature_k}
    assert rows["7"].slope_k_per_ghz is not None and rows["7"].slope_k_per_ghz > 0.1
    assert rows["7"].center_ghz == pytest.approx(325.0)
    assert rows["7"].trx_k == pytest.approx(68.0, rel=0.01)


@pytest.mark.unit
def test_tdm_windows_get_their_own_efficiency():
    """A band's TDM windows over its FDM windows, apart from the band's own scale."""
    fdm = [Measured(_window(k, efficiency_scale=0.95)) for k in range(6)]
    tdm = [Measured(_window(10 + k, efficiency_scale=0.95, tdm=1.05)) for k in range(4)]
    curve = [Measured(_window(20 + k, ghz=100.0 + 60 * k)) for k in range(6)]
    radiometer = aggregate(fdm + tdm + curve, released_model("alma", 3)).components.radiometer
    assert radiometer.tdm_efficiency == pytest.approx(1.05, rel=0.02)
    without = aggregate(fdm + curve, released_model("alma", 3))
    assert without.components.radiometer.tdm_efficiency is None
    assert "radiometer.tdm_efficiency: 0 window(s), kept the base's" in without.provenance.notes


@pytest.mark.unit
def test_too_few_windows_keep_the_base():
    base = released_model("alma")
    candidate = aggregate([Measured(_window(0))], base, name="trial", version=7)
    assert candidate.label() == "trial@7"
    assert candidate.components == base.components
    assert "kept the base's" in candidate.provenance.notes
    # Kept apertures keep the scales relative to them (alma@3 sets both dishes').
    scales = released_model("alma", 3).components.radiometer.efficiency_scale
    kept = aggregate([Measured(_window(0))], released_model("alma", 3)).components.radiometer
    assert scales and kept.efficiency_scale == scales


@pytest.mark.unit
def test_noise_aggregate_writes_the_candidate(tmp_path):
    from almasim.cli.noise import noise_app

    out = tmp_path / "run"
    for k in range(5):
        d = out / "windows" / f"w{k}"
        d.mkdir(parents=True)
        (d / "window.json").write_text(_window(k).model_dump_json())
        (d / "fit.json").write_text(_fit(k).model_dump_json())
    runner = CliRunner()
    result = runner.invoke(noise_app, ["aggregate", "--out", str(out), "--from", "alma@1"])
    assert result.exit_code == 0, result.output
    assert "5 window(s), 5 fitted -> alma-candidate@2" in result.output
    assert "antenna_gains.amp_rms: 0.07" in result.output
    candidate = resolve_noise_model(str(out / "candidate.json"))
    assert candidate.components.antenna_gains.amp_rms == pytest.approx(0.07)
    empty = runner.invoke(noise_app, ["aggregate", "--out", str(tmp_path / "none")])
    assert empty.exit_code == 2 and "run noise extract first" in empty.output
    bad = runner.invoke(noise_app, ["aggregate", "--out", str(out), "--from", "stormy"])
    assert bad.exit_code == 2 and "--from" in bad.output
    assert json.loads((out / "candidate.json").read_text())["provenance"]["windows"] == 5


@pytest.mark.unit
def test_in_scan_strengths_are_brought_to_the_median_coherence():
    """Strength and coherence trade off: each window's strength is taken at the
    median coherence with its phase change over one simulated sample kept."""
    from almasim.services.noise.aggregate import FAST_REFERENCE_S
    from almasim.services.noise.measurements import PhaseFit

    coherences = [10.0, 20.0, 20.0, 40.0, 80.0]
    windows = [
        Measured(
            _window(k),
            _fit(k).model_copy(
                update={
                    "fast_phase": PhaseFit(
                        bins=20,
                        slope=0.5,
                        phase_1km_deg=10.0,
                        path_um_1km=100.0,
                        coherence_s=c,
                    )  # fmt: skip
                }
            ),
        )
        for k, c in enumerate(coherences)
    ]
    fast = aggregate(windows, released_model("alma", 3)).components.fast_phase
    assert fast.coherence_s == 20.0
    change = [100.0**2 * (1 - np.exp(-FAST_REFERENCE_S / c)) for c in coherences]
    expected = np.sqrt(np.median(change) / (1 - np.exp(-FAST_REFERENCE_S / 20.0)))
    assert fast.path_um_1km == pytest.approx(expected)
    assert fast.slope == 0.5


@pytest.mark.unit
def test_each_dish_gets_its_in_scan_strength_and_the_spread_between_observations():
    """The 7 m antennas carry no water vapour radiometers: each dish size with
    enough fitted windows gets its own median, and the windows' log scatter
    about their dish's median sets the spread a simulated observation draws."""
    from almasim.services.noise.measurements import PhaseFit

    paths = {12.0: [40.0, 60.0, 65.0, 80.0, 120.0], 7.0: [130.0, 150.0, 570.0]}
    windows = [
        Measured(
            _window(10 * int(dish) + k, dish=dish),
            _fit(k).model_copy(
                update={
                    "fast_phase": PhaseFit(
                        bins=20, slope=0.5, phase_1km_deg=10.0, path_um_1km=p, coherence_s=20.0
                    )  # fmt: skip
                }
            ),
        )
        for dish, values in paths.items()
        for k, p in enumerate(values)
    ]
    candidate = aggregate(windows, released_model("alma", 4))
    fast = candidate.components.fast_phase
    by_dish = {d.diameter_m: d.path_um_1km for d in fast.dishes}
    assert by_dish == {12.0: 65.0, 7.0: 150.0}
    assert fast.path_um_1km == pytest.approx(np.median(paths[12.0] + paths[7.0]))
    logs = [np.log(p / 65.0) for p in paths[12.0]] + [np.log(p / 150.0) for p in paths[7.0]]
    spread = np.sqrt(np.sum(np.square(logs)) / (len(logs) - 2))
    assert fast.log_spread == pytest.approx(spread)
    values = candidate.provenance.values
    assert values["fast_phase.dishes.7m"].n == 3
    assert values["fast_phase.dishes.7m"].q84 > 150.0
    stats = values["fast_phase.log_spread"]
    assert stats.q16 < spread < stats.q84
    # Too few windows of a dish: it takes path_um_1km, and says so.
    few = aggregate(windows[:7], released_model("alma", 4))
    assert [d.diameter_m for d in few.components.fast_phase.dishes] == [12.0]
    assert "fast_phase.dishes.7m: 2 window(s), takes path_um_1km" in few.provenance.notes


@pytest.mark.unit
def test_a_model_sets_the_tdm_efficiency():
    alma = json.loads(released_model("alma", 3).model_dump_json())
    radiometer = {**alma["components"]["radiometer"], "tdm_efficiency": 1.05}
    model = resolve_noise_model(
        {**alma, "components": {**alma["components"], "radiometer": radiometer}}
    )
    assert model.noise_config()["tdm_efficiency"] == 1.05
    # Unset, it leaves the released models' identity as it is.
    assert "tdm_efficiency" not in released_model("alma", 3).canonical_json()
    assert "tdm_efficiency" not in released_model("alma", 3).noise_config()
