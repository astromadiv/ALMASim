"""The channels of a real window that validation simulates, and the comparison
over them (MADIV-28): an atmospheric line in the simulated channels, line
channels left out alike, and windows whose MS lists channels falling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from almasim.services.noise.channels import (
    SimulatedBand,
    channel_spec,
    real_line_channels,
    simulated_band,
)
from almasim.services.noise.compare import Pair, _value
from almasim.services.noise.extract import parse_channels
from almasim.services.noise.measurements import Tsys, WindowNoise

from .test_noise_validate import _window


def _hot(window: WindowNoise) -> WindowNoise:
    """``window`` with four channels, the central two at four times the
    edges' noise variance in its SYSCAL (an atmospheric line)."""
    tsys = Tsys(median_k=200.0, profile=[1.0, 4.0, 4.0, 1.0], spectral_window=0)
    return window.model_copy(update={"tsys": tsys})


@pytest.mark.unit
def test_a_channel_mask_reads_back_from_its_selection():
    mask = np.zeros(20, dtype=bool)
    mask[[0, 1, 2, 7, 12, 13]] = True
    assert channel_spec(mask) == "0~2;7;12~13"
    np.testing.assert_array_equal(parse_channels(channel_spec(mask), 20), mask)
    assert channel_spec(np.zeros(5, dtype=bool)) == ""


@pytest.mark.unit
def test_a_long_window_is_simulated_over_its_central_channels():
    lines = np.zeros(10, dtype=bool)
    lines[[3, 4]] = True
    band = simulated_band(10, 4, lines=lines)
    assert (band.first, band.count) == (3, 4)
    np.testing.assert_array_equal(np.flatnonzero(band.simulated()), [3, 4, 5, 6])
    # The simulated MS's channels 0 and 1 are the real window's 3 and 4 ...
    assert band.simulated_line_channels() == "0~1"
    # ... or, where the MS lists the window falling, its channels 6 and 5:
    # the simulated window runs in rising frequency.
    assert replace(band, falling=True).simulated_line_channels() == "2~3"
    # A window within the limit is simulated whole.
    assert simulated_band(10, 1024, lines=lines).count == 10


@pytest.mark.unit
def test_the_real_windows_line_channels_come_from_its_selection_or_its_log(tmp_path):
    log = tmp_path / "task.log"
    log.write_text("found lines in channels 2~3;8\n")
    np.testing.assert_array_equal(
        np.flatnonzero(real_line_channels(10, task_log=str(log))), [2, 3, 8]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(real_line_channels(10, line_channels="1", task_log=str(log))), [1]
    )
    assert not real_line_channels(10, task_log=str(tmp_path / "missing.log")).any()


@pytest.mark.unit
def test_the_scales_bring_the_central_channels_to_the_whole_window():
    # Four channels, the central two at four times the edges' noise variance
    # (an atmospheric line); two simulated.
    real = _hot(_window("w", ghz=358.8))
    band = SimulatedBand(total=4, first=1, count=2)
    scales = band.scales(real)
    # Visibilities: the mean variance, 2.5 over the window against 4.
    assert scales["visibility"] == pytest.approx(np.sqrt(2.5 / 4))
    # Continuum, its channels weighed alike: (2/4) x mean variance 2.5 / 4.
    assert scales["band"] == pytest.approx(np.sqrt(2 / 4 * 2.5 / 4))
    assert scales["cube"] == pytest.approx(np.sqrt(2.5 / 4))
    # Line channels are left out of both sides' visibilities.
    lined = SimulatedBand(total=4, first=1, count=2, line_channels="1")
    assert lined.scales(real)["visibility"] == pytest.approx(np.sqrt(2 / 4))
    # A flat or unknown Tsys: the bandwidth alone; a whole window: 1.
    flat = band.scales(_window("w"))
    assert flat == pytest.approx({"visibility": 1.0, "band": np.sqrt(0.5), "cube": 1.0})
    whole = SimulatedBand(total=4, first=0, count=4).scales(real)
    assert whole == pytest.approx({"visibility": 1.0, "band": 1.0, "cube": 1.0})


@pytest.mark.unit
def test_compare_takes_the_simulated_visibilities_to_the_real_channels():
    real = _hot(_window("w", unit=2000.0))
    sim = _window("w", unit=2000.0 * np.sqrt(4 / 2.5))  # noisier: the hot channels
    pair = Pair(name="w", sim_image={}, real_image={}, sim_window=sim, real_window=real)
    assert _value(pair, "visibility.unit_jy", simulated=True) == pytest.approx(
        2000.0 * np.sqrt(4 / 2.5)
    )
    banded = Pair(
        name="w",
        sim_image={},
        real_image={},
        sim_window=sim,
        real_window=real,
        simulated=SimulatedBand(total=4, first=1, count=2),
    )
    assert _value(banded, "visibility.unit_jy", simulated=True) == pytest.approx(2000.0)
    assert _value(banded, "visibility.unit_jy", simulated=False) == pytest.approx(2000.0)
    # A record, not a number, has no value.
    assert _value(banded, "visibility", simulated=True) is None


@pytest.mark.unit
def test_the_simulated_band_is_recorded_and_read_back(tmp_path):
    band = SimulatedBand(total=3840, first=1408, count=1024, falling=True, line_channels="5~9")
    path = tmp_path / "band.json"
    path.write_text(band.to_json())
    assert SimulatedBand.read(path) == band
    assert SimulatedBand.read(tmp_path / "none.json") is None


class _SpectralWindows:
    """A SPECTRAL_WINDOW table: window 0 rising, window 1 falling."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def getcell(self, column, row):
        assert column == "CHAN_WIDTH"
        return np.array([1.0e6, 1.0e6]) * (1 if row == 0 else -1)


@pytest.mark.unit
def test_the_channel_order_is_read_from_the_windows_channel_widths(monkeypatch):
    from almasim.services.imaging import ms_geometry
    from almasim.services.noise.channels import channels_falling

    opened = []

    def table(path, ack=False):
        opened.append(path)
        return _SpectralWindows()

    monkeypatch.setattr(ms_geometry, "import_casacore_tables", lambda: table)
    assert channels_falling("/d/a.ms", 0) == {"falling": False}
    assert channels_falling("/d/a.ms", 1) == {"falling": True}
    assert opened == ["/d/a.ms/SPECTRAL_WINDOW"] * 2


@pytest.mark.unit
def test_a_windows_band_takes_its_order_and_line_channels(monkeypatch, tmp_path):
    from almasim.services.noise import validate as v
    from almasim.services.noise.batch import WindowTask
    from almasim.services.runtime import child

    calls = []

    def call_in_child(target, kwargs, **options):
        calls.append((target, dict(kwargs)))
        return {"falling": True}

    monkeypatch.setattr(child, "call_in_child", call_in_child)
    real = _window("w")
    real = real.model_copy(update={"exposure": real.exposure.model_copy(update={"channels": 3840})})
    task = WindowTask(ms="/d/a.ms", spw=5, name="w", line_channels="1500~1599")
    band = v.window_band(task, real)
    assert calls == [
        ("almasim.services.noise.channels:channels_falling", {"ms": "/d/a.ms", "spw": 5})
    ]
    assert (band.total, band.first, band.count, band.falling) == (3840, 1408, 1024, True)
    # Real channels 1500-1599 are the simulated window's 92-191, numbered
    # from the falling end: 1023 - 191 = 832 to 1023 - 92 = 931.
    assert band.simulated_line_channels() == "832~931"


@pytest.mark.unit
def test_the_simulated_window_is_measured_without_its_line_channels(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from almasim.services.noise import validate as v

    calls = []

    def run(command, capture_output, text):
        calls.append(command)
        out = Path(command[command.index("--out") + 1])
        (out / "windows" / "w").mkdir(parents=True)
        (out / "windows" / "w" / "window.json").write_text(_window("w").model_dump_json())
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(v.subprocess, "run", run)
    v.measure(tmp_path / "sim.ms", "w", tmp_path, line_channels="832~931")
    (command,) = calls
    assert command[command.index("--line-channels") + 1] == "832~931"
