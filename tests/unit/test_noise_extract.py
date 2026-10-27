"""Measuring a real window's noise (MADIV-28): almasim noise extract.

The MS is an in-memory stand-in for python-casacore's tables with known
noise: Hanning-correlated channels (rho1 2/3, rho2 1/6), WEIGHT that matches
the scatter, one antenna with its own noise ripple, a Tsys slope in SYSCAL,
and a sky that channel differences remove.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from almasim.services.noise import batch, extract
from almasim.services.noise.measurements import WindowNoise

NANT, NTIME, NCHAN, SIGMA = 6, 200, 128, 0.5
SPW, TSYS_SPW = 2, 3


class FakeTable:
    """The part of python-casacore's table a measurement reads."""

    def __init__(self, columns: dict[str, np.ndarray]):
        self.columns = columns

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def nrows(self) -> int:
        return len(next(iter(self.columns.values())))

    def colnames(self) -> list[str]:
        return list(self.columns)

    def getcell(self, column: str, row: int):
        return self.columns[column][row]

    def getcol(self, column: str, startrow: int = 0, nrow: int = -1, rowincr: int = 1):
        values = self.columns[column]
        stop = len(values) if nrow < 0 else startrow + nrow * rowincr
        return values[startrow:stop:rowincr]

    @contextmanager
    def query(self, text: str):
        keep = np.ones(self.nrows(), dtype=bool)
        for clause in text.split(" AND "):
            if m := re.fullmatch(r"(\w+) IN \[([\d,]*)\]", clause.strip()):
                keep &= np.isin(self.columns[m[1]], [int(x) for x in m[2].split(",") if x])
            elif m := re.fullmatch(r"(\w+) != (\w+)", clause.strip()):
                keep &= self.columns[m[1]] != self.columns[m[2]]
            elif m := re.fullmatch(r"(\w+) == (\d+)", clause.strip()):
                keep &= self.columns[m[1]] == int(m[2])
            else:
                raise ValueError(clause)
        yield FakeTable({k: v[keep] for k, v in self.columns.items()})


def _fake_ms(seed: int = 0, ripple_antenna: int | None = 0) -> dict[str, FakeTable]:
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(NANT, 1)
    nbl = a1.size
    ant1 = np.tile(a1, NTIME)
    ant2 = np.tile(a2, NTIME)
    # Autocorrelations and another window's rows, which a measurement leaves out.
    nrow = ant1.size
    times = 5.0e9 + 6.0 * np.repeat(np.arange(NTIME), nbl)
    kernel = np.array([1.0, 2.0, 1.0]) / np.sqrt(6.0)  # Hanning: rho1 2/3, rho2 1/6
    white = rng.normal(size=(nrow, NCHAN + 2, 2)) + 1j * rng.normal(size=(nrow, NCHAN + 2, 2))
    noise = SIGMA * sum(kernel[k] * white[:, k : k + NCHAN] for k in range(3))
    if ripple_antenna is not None:
        # Each antenna's noise variance ripples by ~14% (rms), the first one's twice that.
        chan = np.arange(NCHAN)
        phase = rng.uniform(0, 2 * np.pi, NANT)
        amp = np.full(NANT, 0.2)
        amp[ripple_antenna] = 0.4
        g = 1.0 + amp[:, None] * np.sin(2 * np.pi * chan[None, :] / 40.0 + phase[:, None])
        noise *= np.sqrt(g[ant1] * g[ant2])[:, :, None]
    sky = 3.0 + 0.01 * np.arange(NCHAN)[None, :, None] + 0j
    data = (sky + noise).astype(np.complex64)
    flag = np.zeros(data.shape, dtype=bool)
    flag[::17, 5:9] = True
    weight = np.full((nrow, 2), 1.0 / SIGMA**2, dtype=np.float32)
    ddid = np.zeros(nrow, dtype=int)
    field = np.zeros(nrow, dtype=int)
    # One more window and an autocorrelation per time, which must not count.
    extra = NTIME
    columns = {
        "DATA": np.concatenate([data, (100 * data[:extra])]),
        "FLAG": np.concatenate([flag, flag[:extra]]),
        "WEIGHT": np.concatenate([weight, weight[:extra]]),
        "ANTENNA1": np.concatenate([ant1, np.zeros(extra, int)]),
        "ANTENNA2": np.concatenate([ant2, np.zeros(extra, int)]),
        "DATA_DESC_ID": np.concatenate([ddid, np.zeros(extra, int)]),
        "FIELD_ID": np.concatenate([field, np.zeros(extra, int)]),
        "TIME": np.concatenate([times, times[:extra]]),
        "INTERVAL": np.full(nrow + extra, 6.0),
    }
    freq = 230e9 + 15.625e6 * np.arange(NCHAN)
    nspw = 4
    chan_freq = [np.array([1e9 * (k + 1)]) for k in range(nspw)]
    chan_freq[SPW] = freq
    chan_freq[TSYS_SPW] = np.linspace(freq[0] - 0.5e9, freq[-1] + 0.5e9, 64)
    tsys = 100.0 + 20.0 * np.linspace(0, 1, 64)
    tsys_rows = np.broadcast_to(tsys[None, :, None], (NANT, 64, 2)).copy()
    tsys_rows[:, :2] = 2000.0  # rolled-off edges
    return {
        "ms": FakeTable(columns),
        "ms/SPECTRAL_WINDOW": FakeTable(
            {
                "CHAN_FREQ": np.array(chan_freq, dtype=object),
                "CHAN_WIDTH": np.array(
                    [np.full(f.size, 15.625e6) for f in chan_freq], dtype=object
                ),
                "EFFECTIVE_BW": np.array(
                    [np.full(f.size, 2.667 * 15.625e6) for f in chan_freq], dtype=object
                ),
                "BBC_NO": np.array([1, 1, 2, 2]),
            }
        ),
        "ms/DATA_DESCRIPTION": FakeTable(
            {"SPECTRAL_WINDOW_ID": np.array([SPW]), "POLARIZATION_ID": np.array([0])}
        ),
        "ms/POLARIZATION": FakeTable({"CORR_TYPE": np.array([[9, 12]])}),
        "ms/FIELD": FakeTable(
            {"NAME": np.array(["target"]), "PHASE_DIR": np.array([[[1.25, -0.5]]])}
        ),
        "ms/ANTENNA": FakeTable({"DISH_DIAMETER": np.full(NANT, 12.0)}),
        "ms/SYSCAL": FakeTable(
            {"SPECTRAL_WINDOW_ID": np.full(NANT, TSYS_SPW), "TSYS_SPECTRUM": tsys_rows}
        ),
    }


@pytest.fixture
def fake_ms(monkeypatch, tmp_path):
    """Two windows' MSs on disk (as names) whose tables live in memory."""
    stores = {}

    def install(name: str, **kwargs) -> Path:
        path = tmp_path / name
        (path / "SYSCAL").mkdir(parents=True)
        for key, value in _fake_ms(**kwargs).items():
            stores[str(path) + key[2:]] = value
        return path

    def open_table(path: str, ack: bool = False) -> FakeTable:
        return stores[str(path)]

    monkeypatch.setattr(extract, "import_casacore_tables", lambda: open_table)
    return install


@pytest.mark.unit
def test_the_noise_level_correlation_and_tsys_are_measured(fake_ms):
    ms = fake_ms("w.ms.split.cal")
    record, arrays = extract.extract_window(ms, SPW, "target", max_visibilities=10**9)
    assert record.thermal.scale == pytest.approx(1.0, rel=0.05)
    assert record.thermal.rho1 == pytest.approx(2 / 3, abs=0.02)
    assert record.thermal.rho2 == pytest.approx(1 / 6, abs=0.02)
    assert len(record.thermal.channel_profile) == NCHAN
    e = record.exposure
    assert (e.antennas, e.baselines, e.integrations, e.channels) == (NANT, 15, NTIME, NCHAN)
    assert e.time_on_source_s == pytest.approx(6.0 * NTIME)
    assert e.noise_bandwidth_hz == pytest.approx(2.667 * 15.625e6)
    assert e.elevation_deg is not None and -90 <= e.elevation_deg <= 90
    assert 0 < e.flagged_fraction < 0.02
    # Stokes I from independent hands: sigma / sqrt(2) per component.
    assert record.visibility.sigma_jy == pytest.approx(SIGMA / np.sqrt(2), rel=0.05)
    assert record.visibility.unit_jy == pytest.approx(
        record.visibility.sigma_jy * np.sqrt(6.0 * 15.625e6)
    )
    assert not record.visibility.hands_identical
    # SYSCAL: the Tsys slope across the window, its edges left out.
    assert record.tsys is not None and record.tsys.spectral_window == TSYS_SPW
    assert 100 < record.tsys.median_k < 120
    assert record.tsys.profile[-1] > record.tsys.profile[0]
    assert record.visibility.efficiency is not None and record.visibility.efficiency > 0
    # The antenna with its own ripple shows it; the shape's arrays cover the window.
    assert record.antenna_shape is not None and record.antenna_shape.excess_rms > 0.08
    assert arrays["antenna_g"].shape == (NANT, NCHAN, 2)
    assert np.ptp(arrays["antenna_g"][0, :, 0]) > np.ptp(arrays["antenna_g"][3, :, 0])
    assert record.notes == []


@pytest.mark.unit
def test_sampling_line_channels_and_selection(fake_ms):
    ms = fake_ms("w.ms")
    lines = extract.parse_channels("20~40;100", NCHAN)
    assert lines.sum() == 22
    sampled, _ = extract.extract_window(ms, SPW, 0, line_channels=lines, max_visibilities=200_000)
    assert sampled.thermal.row_step > 1 and sampled.input.line_channels == 22
    assert sampled.thermal.scale == pytest.approx(1.0, rel=0.08)
    with pytest.raises(ValueError, match="no field named"):
        extract.extract_window(ms, SPW, "sgr_a")
    with pytest.raises(ValueError, match="no spectral window 9"):
        extract.extract_window(ms, 9)
    with pytest.raises(ValueError, match="no data description"):
        extract.extract_window(ms, TSYS_SPW)
    with pytest.raises(ValueError, match="at least 8"):
        extract.extract_window(ms, 0)
    plain = fake_ms("plain.ms", ripple_antenna=None)
    record, _ = extract.extract_window(plain, SPW)
    assert record.antenna_shape is not None and record.antenna_shape.excess_rms < 0.03


@pytest.mark.unit
def test_line_channels_come_from_an_imaging_log(tmp_path):
    log = tmp_path / "task.log"
    log.write_text("x\nINFO uv-contsub: lines in channels 3~5;9\n")
    assert np.flatnonzero(extract.line_channels_from_log(log, 12)).tolist() == [3, 4, 5, 9]
    log.write_text("lines in channels none\n")
    assert not extract.line_channels_from_log(log, 12).any()


@pytest.mark.unit
def test_windows_are_measured_once_and_failures_kept(fake_ms, tmp_path):
    from almasim.cli.noise import noise_app

    first, second = fake_ms("a.ms.split.cal"), fake_ms("b.ms.split.cal", seed=1)
    table = tmp_path / "windows.csv"
    table.write_text(
        f"ms,spw,field,line_channels\n{first},{SPW},target,10~12\n{second},{SPW},,\n{second},9,,\n"
    )
    names = [t.name for t in batch.read_windows(table)]
    assert names == [f"a_spw{SPW}_fieldtarget", f"b_spw{SPW}", "b_spw9"]
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(noise_app, ["extract", "--windows", str(table), "--out", str(out)])
    assert result.exit_code == 1, result.output
    failures = json.loads((out / "failures.json").read_text())
    assert list(failures) == ["b_spw9"] and "no spectral window 9" in failures["b_spw9"]
    measured = WindowNoise.model_validate_json(
        (out / "windows" / f"a_spw{SPW}_fieldtarget" / "window.json").read_text()
    )
    assert measured.input.line_channels == 3 and measured.input.field == "target"
    assert (out / "windows" / f"b_spw{SPW}" / "antenna.npz").is_file()
    again = runner.invoke(noise_app, ["extract", "--windows", str(table), "--out", str(out)])
    assert "2 window(s) already measured; 1 to go" in again.output

    single = runner.invoke(
        noise_app, ["extract", str(first), "--spw", str(SPW), "--out", str(tmp_path / "one")]
    )
    assert single.exit_code == 0, single.output
    assert (tmp_path / "one" / "windows" / f"a_spw{SPW}" / "window.json").is_file()


@pytest.mark.unit
def test_extract_checks_its_arguments(tmp_path):
    from almasim.cli.noise import noise_app

    runner = CliRunner()
    cases = [
        ([], "give an MS"),
        (["x.ms"], "--spw is needed"),
        (["--windows", "w.csv", "--spw", "3"], "describe one MS"),
        (["--windows", str(tmp_path / "missing.csv")], "--windows"),
    ]
    for args, message in cases:
        result = runner.invoke(noise_app, ["extract", *args, "--out", str(tmp_path)])
        assert result.exit_code == 2 and message in result.output, (args, result.output)
    bad = tmp_path / "bad.csv"
    bad.write_text("ms\nx.ms\n")
    with pytest.raises(ValueError, match="needs column"):
        batch.read_windows(bad)
    bad.write_text("ms,spw,name\nx.ms,1,a\ny.ms,2,a\n")
    with pytest.raises(ValueError, match="repeat"):
        batch.read_windows(bad)


@pytest.mark.unit
def test_a_window_task_runs_as_the_same_command(tmp_path):
    task = batch.WindowTask(ms="/d/x.ms", spw=3, name="x", field="2", line_channels="1~4")
    assert task.arguments() == [
        "/d/x.ms", "--spw", "3", "--name", "x", "--field", "2", "--line-channels", "1~4",
    ]  # fmt: skip
    assert batch.window_name("/d/uid___A002_X1.ms.split.cal", 31, None) == "uid___A002_X1_spw31"


@pytest.mark.unit
def test_windows_fan_out_as_commands_on_slurm(monkeypatch, tmp_path):
    """On Slurm each window is its own `almasim noise extract`; a failed task's
    stderr is its failure."""
    from types import SimpleNamespace

    import almasim.services.compute as compute

    submitted = []

    class Cluster:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit_subcommand(self, *, command, cores, timeout):
            submitted.append((command, cores, timeout))
            failed = "bad" in command
            result = SimpleNamespace(
                returncode=1 if failed else 0, stderr="boom\n" if failed else ""
            )
            return SimpleNamespace(result=lambda: result)

    seen = {}

    def create_backend(kind, **kwargs):
        seen.update(kind=kind, **kwargs)
        return Cluster()

    monkeypatch.setattr(compute, "create_backend", create_backend)
    tasks = [
        batch.WindowTask(ms="/d/a.ms", spw=1, name="a"),
        batch.WindowTask(ms="/d/b.ms", spw=2, name="bad"),
    ]
    results = batch.run_on_slurm(
        tasks,
        out=tmp_path,
        backend_kwargs={"n_workers": 8, "queue": "q"},
        cores=2,
        timeout=60.0,
        extra_args=["--overwrite"],
    )
    assert results == {"a": None, "bad": "boom"}
    assert seen == {"kind": "slurm", "n_workers": 2, "queue": "q"}
    command, cores, timeout = submitted[0]
    assert command[-9:] == [
        "extract",
        "/d/a.ms",
        "--spw",
        "1",
        "--name",
        "a",
        "--out",
        str(tmp_path),
        "--overwrite",
    ]
    assert (cores, timeout) == (2, 60.0)
