"""The report of what ``noise extract`` measured: every metric explained, every window's values."""

from __future__ import annotations

import csv
import json

import pytest
from typer.testing import CliRunner

from almasim import cli
from almasim.services.noise.measurements import (
    AntennaNoise,
    AntennaShape,
    Exposure,
    ThermalNoise,
    Tsys,
    VisibilityNoise,
    WindowInput,
    WindowNoise,
)
from almasim.services.noise.report import (
    DATA_FILE,
    REPORT_FILE,
    TABLE_FILE,
    metrics,
    spread,
    window_values,
    write_extract_report,
)

runner = CliRunner()


def _window(name: str, scale: float, tsys: bool = True) -> WindowNoise:
    return WindowNoise(
        name=name,
        almasim_version="test",
        input=WindowInput(ms=f"/d/{name}.ms", spw=0, pwv_mm=1.2),
        exposure=Exposure(
            antennas=40,
            dish_diameter_m=12.0,
            baselines=780,
            integrations=100,
            integration_s=6.0,
            time_on_source_s=600.0,
            channels=4,
            channel_width_hz=15.6e6,
            noise_bandwidth_hz=41.7e6,
            frequency_ghz=230.0,
            correlations=2,
            flagged_fraction=0.01,
        ),  # fmt: skip
        thermal=ThermalNoise(
            scale=scale,
            rho1=0.66,
            rho2=0.16,
            channel_profile=[1.1, 0.9, 1.1, 0.9],
            rows_sampled=10,
            row_step=1,
        ),
        visibility=VisibilityNoise(
            sigma_jy=1.0, unit_jy=2000.0, hands_identical=False, efficiency=0.6
        ),
        tsys=Tsys(median_k=90.0, profile=[1.0] * 4, spectral_window=17) if tsys else None,
        antenna_shape=AntennaShape(
            antennas_used=40,
            channels_used=4,
            excess_rms=0.03,
            half_width_channels=2,
            half_width_mhz=31.2,
        ),
        antenna_noise=AntennaNoise(antennas_used=40, baselines_used=780, sefd_log_rms=0.1 * scale),
        notes=[] if tsys else ["no SYSCAL table: Tsys not measured"],
    )


def _extraction(tmp_path, windows):
    for window in windows:
        folder = tmp_path / "windows" / window.name
        folder.mkdir(parents=True)
        (folder / "window.json").write_text(window.model_dump_json())
    (tmp_path / "failures.json").write_text(json.dumps({"w9": "RuntimeError: no rows"}))
    return tmp_path


@pytest.mark.unit
def test_every_measured_quantity_has_a_meaning_and_a_unit():
    catalogue = {m.key: m for m in metrics()}
    assert "thermal.scale" in catalogue and "visibility.total_power.sefd_ratio" in catalogue
    assert "exposure.pads" not in catalogue and "notes" not in catalogue
    for metric in catalogue.values():
        assert metric.meaning, metric.key
        if metric.measured and metric.kind == "number":
            assert metric.unit, metric.key
    assert catalogue["thermal.channel_profile"].kind == "profile"
    assert catalogue["visibility.hands_identical"].kind == "flag"
    assert not catalogue["input.ms"].measured


@pytest.mark.unit
def test_a_profile_is_reported_as_its_spread_about_one():
    values = window_values(_window("w1", 1.0))
    assert values["thermal.channel_profile"] == pytest.approx(0.1)
    assert values["tsys.profile"] == pytest.approx(0.0)
    assert values["visibility.total_power.sefd_ratio"] is None  # not measured


@pytest.mark.unit
def test_the_spread_leaves_out_windows_without_the_metric():
    found = spread([1.0, None, 3.0, 2.0])
    assert (found.windows, found.median) == (3, 2.0)
    assert spread([None]).windows == 0


@pytest.mark.unit
def test_the_report_explains_each_metric_and_lists_each_window(tmp_path):
    out = _extraction(tmp_path, [_window("w1", 1.0), _window("w2", 1.4, tsys=False)])
    lines: list[str] = []

    done = write_extract_report(out, echo=lines.append)

    assert done.windows == 2
    report = (out / REPORT_FILE).read_text()
    assert "2 window(s) measured, 1 failed" in report
    assert "## Thermal noise against WEIGHT" in report
    row = next(line for line in report.splitlines() if line.startswith("| `thermal.scale`"))
    assert "| ratio | 1.2 | 1.064 - 1.336 | 2 |" in row
    assert "WEIGHT column states the noise right" in row
    tsys = next(line for line in report.splitlines() if line.startswith("| `tsys.median_k`"))
    assert "| K | 90 | 90 - 90 | 1 |" in tsys
    assert "- **w2**: no SYSCAL table: Tsys not measured" in report
    assert "- **w9**: RuntimeError: no rows" in report
    assert "## Each antenna's own noise level" in report
    level = next(
        line for line in report.splitlines() if line.startswith("| `antenna_noise.sefd_log_rms`")
    )
    assert "| ln | 0.12 |" in level
    assert "| w2 | 230 | 40 |" in report

    with (out / TABLE_FILE).open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["name"] for r in rows] == ["w1", "w2"]
    assert rows[1]["thermal.scale"] == "1.4" and rows[1]["tsys.median_k"] == ""
    assert rows[1]["notes"] == "no SYSCAL table: Tsys not measured"
    data = json.loads((out / DATA_FILE).read_text())
    thermal = next(s for s in data["sections"] if s["key"] == "thermal")
    scale = next(m for m in thermal["metrics"] if m["key"] == "thermal.scale")
    assert (scale["median"], scale["windows"], scale["unit"]) == (1.2, 2, "ratio")
    assert data["rows"][1]["tsys.median_k"] is None
    assert data["failures"] == {"w9": "RuntimeError: no rows"}
    assert any(line.lstrip().startswith("thermal.scale") for line in lines)
    assert lines[-2].startswith("Report: ")


@pytest.mark.unit
def test_noise_report_writes_the_report_of_an_extraction_folder(tmp_path):
    out = _extraction(tmp_path, [_window("w1", 1.0)])

    result = runner.invoke(cli.app, ["noise", "report", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / REPORT_FILE).is_file() and (out / TABLE_FILE).is_file()

    result = runner.invoke(cli.app, ["noise", "report", "--out", str(tmp_path / "none")])
    assert result.exit_code == 1


@pytest.mark.unit
def test_noise_extract_writes_its_report(tmp_path, monkeypatch):
    from almasim.services.noise import batch

    def _measure(task, out, **options):
        window = _window(task.name, 1.0)
        folder = task.directory(out)
        folder.mkdir(parents=True)
        path = folder / "window.json"
        path.write_text(
            window.model_copy(
                update={"input": WindowInput(ms=task.ms, spw=task.spw)}
            ).model_dump_json()
        )
        return path

    monkeypatch.setattr(batch, "measure_task", _measure)
    result = runner.invoke(
        cli.app,
        ["noise", "extract", "/d/x.ms.split.cal", "--spw", "3", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "Report: " in result.output
    assert (tmp_path / REPORT_FILE).is_file()
