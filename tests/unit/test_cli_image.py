"""Unit tests for ALMASim image CLI commands."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from almasim import cli
from almasim.cli import image as cli_image
from almasim.services.imaging import batch_runner
from almasim.services.imaging import markers as imaging_markers
from almasim.services.imaging import products as imaging_products
from almasim.services.imaging import task as imaging_task
from almasim.services.imaging.ms import single_window as imaging_ms_single_window
from almasim.services.ms.tables import intent_matches
from almasim.services.runtime import env as runtime_env

from ._fakes import FakeTable, flat_run_kwargs, patch_everywhere

runner = CliRunner()


@pytest.mark.unit
def test_import_casacore_tables_returns_table_symbol(monkeypatch):
    """import_casacore_tables should return casacore.tables.table on success."""
    fake_table_symbol = object()
    casacore_module = ModuleType("casacore")
    tables_module = ModuleType("casacore.tables")
    tables_module.table = fake_table_symbol
    casacore_module.tables = tables_module

    monkeypatch.setitem(sys.modules, "casacore", casacore_module)
    monkeypatch.setitem(sys.modules, "casacore.tables", tables_module)

    table_symbol = cli_image.import_casacore_tables()

    assert table_symbol is fake_table_symbol


@pytest.mark.unit
def test_import_casacore_tables_raises_when_casacore_missing(monkeypatch):
    """import_casacore_tables should raise a friendly message if casacore is unavailable."""
    casacore_module = ModuleType("casacore")
    monkeypatch.setitem(sys.modules, "casacore", casacore_module)
    monkeypatch.delitem(sys.modules, "casacore.tables", raising=False)

    with pytest.raises(cli_image.MissingDependencyError, match="python-casacore"):
        cli_image.import_casacore_tables()


@pytest.mark.unit
def test_compute_imaging_parameters_builds_expected_dataframe(monkeypatch):
    """compute_imaging_parameters should populate expected imaging columns."""

    spectral_window = FakeTable(
        {
            "REF_FREQUENCY": np.array([100.0e9, 200.0e9]),
            "NAME": np.array(["X1#BB_1#SW-01#FULL_RES", "X1#BB_2#SW-01#FULL_RES"]),
            "NUM_CHAN": np.array([4, 2]),
            "CHAN_WIDTH": [np.full(4, 0.5e6), np.full(2, -15.625e6)],
            "RESOLUTION": [np.full(4, 1.0e6), np.full(2, -31.25e6)],
            "TOTAL_BANDWIDTH": np.array([2.0e6, 31.25e6]),
        }
    )
    antenna = FakeTable(
        {
            "DISH_DIAMETER": np.array([12.0, 10.0]),
            "POSITION": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0],
                    [0.0, 0.0, 12.0],
                ]
            ),
        }
    )

    fake_observation = [
        {
            "TIME_RANGE": [57000 * 86400.0, 57001 * 86400.0],
        }
    ]

    def _fake_casacore_table(table_name: str, ack: bool = False):
        del ack
        if table_name.endswith("::SPECTRAL_WINDOW"):
            return spectral_window
        if table_name.endswith("::ANTENNA"):
            return antenna
        if table_name.endswith("::OBSERVATION"):
            return fake_observation
        if table_name.endswith("::FIELD"):
            return FakeTable({"NAME": np.array(["cal", "x", "target"])})

    patch_everywhere(monkeypatch, cli_image, "import_casacore_tables", lambda: _fake_casacore_table)
    patch_everywhere(
        monkeypatch,
        cli_image,
        "_field_scan",
        lambda _ms, _intents=(): ({0: 15, 1: 20}, [2], {(0, 0): 5, (0, 2): 10, (1, 2): 20}),
    )

    output = cli_image.compute_imaging_parameters(Path("test_dataset.cal"))
    assert list(output["n_visibility_rows"]) == [10, 20], "the calibrator (field 0) is left out"
    assert list(output["target_field_ids"]) == ["2", "2"]
    assert list(output["field_name"]) == ["target", "target"]
    assert list(output["spectral_window_name"]) == [
        "X1#BB_1#SW-01#FULL_RES",
        "X1#BB_2#SW-01#FULL_RES",
    ]
    assert list(output["n_channels"]) == [4, 2]
    assert list(output["channel_width_hz"]) == [0.5e6, 15.625e6], "widths are absolute"
    assert list(output["spectral_resolution_hz"]) == [1.0e6, 31.25e6]

    expected_frequencies = np.array([100.0e9, 200.0e9])
    speed_of_light = 299_792_458.0
    radians_to_arcsec = 180.0 * 3600.0 / np.pi
    expected_max_baseline_size = 13.0
    expected_wavelengths = speed_of_light / expected_frequencies
    # The primary beam's half-power width (Airy model); the 10 m dish is unblocked.
    from almasim.services.imaging.primary_beam import airy_fwhm_factor

    smallest = np.min(antenna.getcol("DISH_DIAMETER"))
    expected_fov_per_frequency = (
        airy_fwhm_factor(float(smallest)) * expected_wavelengths / smallest * radians_to_arcsec
    )
    expected_synthetized_beam_size = (
        expected_wavelengths / expected_max_baseline_size * radians_to_arcsec
    )

    assert list(output["filename"]) == [str(Path("test_dataset.cal").resolve())] * 2
    np.testing.assert_array_equal(output["spectral_window_id"].to_numpy(), np.array([0, 1]))
    np.testing.assert_array_equal(output["reference_frequency"].to_numpy(), expected_frequencies)
    np.testing.assert_allclose(
        output["max_baseline_size"].to_numpy(),
        np.array([expected_max_baseline_size, expected_max_baseline_size]),
    )
    np.testing.assert_allclose(
        output["fov_per_frequency"].to_numpy(),
        expected_fov_per_frequency,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        output["synthetized_beam_size"].to_numpy(),
        expected_synthetized_beam_size,
        rtol=1e-12,
    )


@pytest.mark.unit
def test_imaging_parameter_to_command_arg_returns_expected_tokens():
    """Command arg helper should return split tokens ready for subprocess usage."""
    params = {
        "spectral_window_id": 3,
        "fov_per_frequency": 8.0,
        "synthetized_beam_size": 2.0,
    }

    cmd_args = cli_image.imaging_parameter_to_command_arg(
        params,
        fov_fraction=1.5,
        beam_sampling=2,
    )

    assert cmd_args[:7] == ["-scale", "1.0asec", "-size", "16", "16", "-spws", "3"]


@pytest.mark.unit
def test_ms_overview_command_prints_dataframe(monkeypatch):
    """ms-overview should print the computed dataframe."""
    df = pd.DataFrame(
        {
            "filename": ["a.cal"],
            "spectral_window_id": [0],
            "reference_frequency": [100.0],
            "fov_per_frequency": [10.0],
            "max_baseline_size": [50.0],
            "synthetized_beam_size": [1.0],
        }
    )
    monkeypatch.setattr(cli_image, "compute_imaging_parameters", lambda _: df)

    result = runner.invoke(cli.app, ["image", "ms-overview", "a.cal"])

    assert result.exit_code == 0
    assert "filename" in result.output
    assert "synthetized_beam_size" in result.output


@pytest.mark.unit
def test_ms_overview_snake_case_command_is_rejected():
    """Only hyphenated command naming should be supported."""
    result = runner.invoke(cli.app, ["image", "ms_overview", "a.cal"])

    assert result.exit_code != 0
    assert "No such command" in result.output


@pytest.mark.unit
def test_compute_parameters_exits_when_no_ms_found(tmp_path):
    """compute-parameters should fail with exit code 1 if no .cal datasets exist."""
    out_csv = tmp_path / "imaging_parameters.csv"

    result = runner.invoke(
        cli.app,
        ["image", "compute-parameters", str(tmp_path), str(out_csv)],
    )

    assert result.exit_code == 1
    assert "Cannot find any MS" in result.output


@pytest.mark.unit
def test_compute_parameters_writes_csv_for_all_datasets(monkeypatch, tmp_path):
    """compute-parameters should aggregate rows across all matching datasets."""
    (tmp_path / "first.cal").mkdir()
    (tmp_path / "second.cal").mkdir()
    out_csv = tmp_path / "imaging_parameters.csv"

    patch_everywhere(monkeypatch, cli_image, "tqdm", lambda iterable: iterable)

    def _fake_compute(input_ms: Path, science_only: bool = True, intents=None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "filename": [str(input_ms.resolve())],
                "spectral_window_id": [0],
                "reference_frequency": [100.0],
                "fov_per_frequency": [10.0],
                "max_baseline_size": [50.0],
                "synthetized_beam_size": [1.0],
            }
        )

    monkeypatch.setattr(cli_image, "compute_imaging_parameters", _fake_compute)

    result = runner.invoke(
        cli.app,
        ["image", "compute-parameters", str(tmp_path), str(out_csv)],
    )

    assert result.exit_code == 0
    saved = pd.read_csv(out_csv)
    assert len(saved) == 2
    assert sorted(Path(f).name for f in saved["filename"].tolist()) == ["first.cal", "second.cal"]


@pytest.mark.unit
def test_compute_parameters_defaults_to_cwd_and_default_output(monkeypatch, tmp_path):
    """compute-parameters should run without args using cwd and default output filename."""
    patch_everywhere(monkeypatch, cli_image, "tqdm", lambda iterable: iterable)

    def _fake_compute(input_ms: Path, science_only: bool = True, intents=None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "filename": [str(input_ms.resolve())],
                "spectral_window_id": [0],
                "reference_frequency": [100.0],
                "fov_per_frequency": [10.0],
                "max_baseline_size": [50.0],
                "synthetized_beam_size": [1.0],
            }
        )

    monkeypatch.setattr(cli_image, "compute_imaging_parameters", _fake_compute)

    monkeypatch.chdir(tmp_path)
    Path("sample.cal").mkdir()
    result = runner.invoke(cli.app, ["image", "compute-parameters"])

    assert result.exit_code == 0
    output_candidates = list(tmp_path.rglob("imaging_parameters.csv"))
    assert output_candidates


@pytest.mark.unit
def test_run_commands_with_slurm_cluster_no_commands_is_noop(monkeypatch):
    """Slurm command runner should return immediately when no commands are provided."""
    called = {"create_backend": False}

    def _fake_create_backend(*args, **kwargs):
        del args, kwargs
        called["create_backend"] = True
        raise AssertionError("create_backend should not be called for an empty command list")

    monkeypatch.setattr("almasim.services.compute.create_backend", _fake_create_backend)

    cli_image._run_commands_with_slurm_cluster(
        commands=[],
        cores_per_task=1,
        node_cores=2,
        queue="normal",
        project=None,
        walltime="00:10:00",
        memory="2GB",
        n_jobs=1,
        scheduler_host=None,
        scheduler_interface=None,
        task_timeout=None,
    )

    assert called["create_backend"] is False


@pytest.mark.unit
def test_predict_from_model_exits_when_output_exists(tmp_path):
    """predict_from_model should fail fast when the output MS already exists."""
    input_ms = tmp_path / "input.ms"
    output_ms = tmp_path / "output.ms"
    model = tmp_path / "model.fits"
    input_ms.mkdir()
    output_ms.mkdir()
    model.write_text("dummy", encoding="utf-8")

    with pytest.raises(typer.Exit) as exc:
        cli_image.predict_from_model(input_ms=input_ms, model=model, output_ms=output_ms)

    assert exc.value.exit_code == 1


@pytest.mark.unit
def test_run_commands_with_slurm_cluster_forwards_scheduler_interface(monkeypatch):
    """Slurm command runner should pass scheduler_interface to backend creation."""
    captured: dict[str, object] = {}

    class _Future:
        def result(self):
            return SimpleNamespace(returncode=0, stderr="")

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def submit_subcommand(self, **kwargs):
            captured["submit_kwargs"] = kwargs
            return _Future()

    def _fake_create_backend(*args, **kwargs):
        captured["create_args"] = args
        captured["create_kwargs"] = kwargs
        return _Backend()

    monkeypatch.setattr("almasim.services.compute.create_backend", _fake_create_backend)
    patch_everywhere(monkeypatch, cli_image, "tqdm", lambda iterable, **kwargs: iterable)

    cli_image._run_commands_with_slurm_cluster(
        commands=[("job-1", ["echo", "ok"])],
        cores_per_task=2,
        node_cores=8,
        queue="normal",
        project=None,
        walltime="00:10:00",
        memory="4GB",
        n_jobs=1,
        scheduler_host="headnode",
        scheduler_interface="ib0",
        task_timeout=10.0,
    )

    assert captured["create_kwargs"]["scheduler_interface"] == "ib0"


@pytest.mark.unit
def test_batch_image_submits_commands_via_slurm_cluster(monkeypatch, tmp_path):
    """batch-image should build one WSClean task per row and hand them to the runner."""
    imaging_csv = tmp_path / "imaging_parameters.csv"
    output_dir = tmp_path / "images"
    output_dir.mkdir()

    pd.DataFrame(
        {
            "filename": ["uid___A001_X1_X1.cal"],
            "spectral_window_id": [2],
            "reference_frequency": [100.0e9],
            "fov_per_frequency": [8.0],
            "max_baseline_size": [100.0],
            "synthetized_beam_size": [2.0],
            "target_field_ids": ["3,4"],
        }
    ).to_csv(imaging_csv, index=False)

    captured: dict[str, Any] = {}

    def _fake_run_imaging_tasks(tasks, imaging, execution, **kwargs):
        captured["tasks"] = tasks
        captured.update(flat_run_kwargs(imaging, execution, **kwargs))
        return []

    monkeypatch.setattr(cli_image, "run_imaging_tasks", _fake_run_imaging_tasks)

    result = runner.invoke(
        cli.app,
        [
            "image",
            "batch-image",
            str(imaging_csv),
            str(output_dir),
            "--num-cores",
            "8",
            "--max-cores-per-node",
            "64",
            "--wsclean-bin",
            "/opt/wsclean/bin/wsclean",
            "--no-single-window-ms",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["single_window"] is False
    assert captured["cores_per_task"] == 8
    assert captured["node_cores"] == 64
    assert captured["queue"] == "normal"
    assert captured["project"] is None
    assert captured["walltime"] == "02:00:00"
    assert captured["memory"] == "16GB"
    assert captured["n_jobs"] == 1
    assert captured["backend_kind"] == "slurm"
    assert captured["continue_on_error"] is True

    tasks = captured["tasks"]
    assert len(tasks) == 1
    task = tasks[0]
    assert task.spw == 2
    assert task.output_dir == output_dir / "uid___A001_X1_X1" / "SPW-2"
    cmd = task.command
    assert cmd[0] == "/opt/wsclean/bin/wsclean"
    assert cmd[cmd.index("-j") + 1] == "8"
    assert cmd[cmd.index("-field") + 1] == "3,4"
    assert cmd[cmd.index("-spws") + 1] == "2"
    assert Path(cmd[-1]).is_absolute() and Path(cmd[-1]).name == "uid___A001_X1_X1.cal"
    assert cmd[cmd.index("-auto-mask") + 1] == "3.0", "masked clean by default"
    assert cmd[cmd.index("-auto-threshold") + 1] == "1.0", "down to 1 sigma inside the mask"
    assert cmd[cmd.index("-size") + 1] == "256", "6 beams of FOV widened to 32 beams x 8 px"
    assert "-multiscale" not in cmd
    assert cmd[cmd.index("-mem") + 1] == "12.5", "-mem is a percentage: 8 of 64 cores"
    assert cmd[cmd.index("-temp-dir") + 1] == str(task.output_dir)
    assert "-no-update-model-required" in cmd and "-update-model-required" not in cmd


@pytest.mark.unit
def test_batch_image_enforces_positive_max_cores_per_node(tmp_path):
    """batch-image should reject invalid max cores value at CLI parsing time."""
    imaging_csv = tmp_path / "imaging_parameters.csv"
    output_dir = tmp_path / "images"
    output_dir.mkdir()

    pd.DataFrame(
        {
            "filename": ["uid___A001_X1_X1.cal"],
            "spectral_window_id": [2],
            "reference_frequency": [100.0e9],
            "fov_per_frequency": [8.0],
            "max_baseline_size": [100.0],
            "synthetized_beam_size": [2.0],
        }
    ).to_csv(imaging_csv, index=False)

    result = runner.invoke(
        cli.app,
        [
            "image",
            "batch-image",
            str(imaging_csv),
            str(output_dir),
            "--max-cores-per-node",
            "0",
        ],
    )

    assert result.exit_code == 2


@pytest.mark.unit
def test_predict_batch_submits_jobs_via_slurm_cluster(monkeypatch, tmp_path):
    """predict-batch should submit predict-single commands via SLURM cluster helper."""
    imaging_csv = tmp_path / "imaging_parameters.csv"
    output_dir = tmp_path / "images"
    output_dir.mkdir()

    first_ms = tmp_path / "uid___A001_X1_X1.cal"
    second_ms = tmp_path / "uid___A001_X2_X2.cal"
    pd.DataFrame(
        {
            "filename": [str(first_ms), str(second_ms)],
            "spectral_window_id": [1, 3],
            "reference_frequency": [100.0e9, 101.0e9],
            "fov_per_frequency": [8.0, 8.2],
            "max_baseline_size": [100.0, 100.0],
            "synthetized_beam_size": [2.0, 2.1],
        }
    ).to_csv(imaging_csv, index=False)

    first_model = output_dir / first_ms.stem / "SPW-1" / "wsclean-model.fits"
    second_model = output_dir / second_ms.stem / "SPW-3" / "wsclean-model.fits"
    first_model.parent.mkdir(parents=True)
    second_model.parent.mkdir(parents=True)
    first_model.touch()
    second_model.touch()

    captured: dict[str, Any] = {}

    def _fake_run_with_slurm_cluster(
        commands,
        *,
        cores_per_task,
        node_cores,
        queue,
        project,
        walltime,
        memory,
        n_jobs,
        scheduler_host,
        scheduler_interface,
        task_timeout,
    ):
        captured["commands"] = commands
        captured["cores_per_task"] = cores_per_task
        captured["node_cores"] = node_cores
        captured["queue"] = queue
        captured["project"] = project
        captured["walltime"] = walltime
        captured["memory"] = memory
        captured["n_jobs"] = n_jobs
        captured["scheduler_host"] = scheduler_host
        captured["scheduler_interface"] = scheduler_interface
        captured["task_timeout"] = task_timeout

    monkeypatch.setattr(cli_image, "_run_commands_with_slurm_cluster", _fake_run_with_slurm_cluster)
    patch_everywhere(monkeypatch, cli_image, "tqdm", lambda iterable, total=None: iterable)

    result = runner.invoke(
        cli.app,
        ["image", "predict-batch", str(imaging_csv), str(output_dir), "--use-slurm"],
    )

    assert result.exit_code == 0
    assert captured["cores_per_task"] == 1
    assert captured["node_cores"] == 95
    assert captured["queue"] == "normal"
    assert captured["project"] is None
    assert captured["walltime"] == "02:00:00"
    assert captured["memory"] == "16GB"
    assert captured["n_jobs"] == 1

    commands = captured["commands"]
    assert len(commands) == 2
    first_cmd = commands[0][1]
    second_cmd = commands[1][1]
    assert first_cmd[:3] == ["almasim", "image", "predict-single"]
    assert second_cmd[:3] == ["almasim", "image", "predict-single"]
    assert str(first_model) in first_cmd
    assert str(second_model) in second_cmd
    assert str(first_model.parent / f"{first_ms.name}.predicted") in first_cmd
    assert str(second_model.parent / f"{second_ms.name}.predicted") in second_cmd


@pytest.mark.unit
def test_predict_batch_skips_when_model_is_missing(monkeypatch, tmp_path):
    """predict-batch should log and skip rows whose model FITS file is missing."""
    imaging_csv = tmp_path / "imaging_parameters.csv"
    output_dir = tmp_path / "images"
    output_dir.mkdir()

    ms_path = tmp_path / "uid___A001_X9_X9.cal"
    pd.DataFrame(
        {
            "filename": [str(ms_path)],
            "spectral_window_id": [0],
            "reference_frequency": [100.0e9],
            "fov_per_frequency": [8.0],
            "max_baseline_size": [100.0],
            "synthetized_beam_size": [2.0],
        }
    ).to_csv(imaging_csv, index=False)

    calls: list[tuple[list[str], bool]] = []

    def _fake_subprocess_run(cmd: list[str], check: bool = False):
        calls.append((cmd, check))

    patch_everywhere(monkeypatch, cli_image, "tqdm", lambda iterable, total=None: iterable)
    monkeypatch.setattr(cli_image.subprocess, "run", _fake_subprocess_run)

    result = runner.invoke(
        cli.app,
        ["image", "predict-batch", str(imaging_csv), str(output_dir)],
    )

    assert result.exit_code == 0
    assert "Missing model FITS, skipping row" in result.output
    # The values are filled in (the message used to print "{input_filename}").
    assert "{input_filename}" not in result.output
    assert f"ms={ms_path}" in result.output
    assert len(calls) == 0


# ---------------------------------------------------------------------------
# Imaging markers, the worker-side WSClean wrapper and the resilient runner.


def _params_csv(tmp_path: Path, rows: int = 2) -> Path:
    csv_path = tmp_path / "params.csv"
    pd.DataFrame(
        {
            "filename": ["/data/uid___A001_X1_X1.ms.split.cal"] * rows,
            "spectral_window_id": list(range(rows)),
            "reference_frequency": [100.0e9] * rows,
            "fov_per_frequency": [8.0] * rows,
            "max_baseline_size": [100.0] * rows,
            "synthetized_beam_size": [2.0] * rows,
        }
    ).to_csv(csv_path, index=False)
    return csv_path


class _FakeWsclean:
    """Stand-in for ``subprocess.Popen`` running WSClean.

    ``returncode`` drives the outcome; on success the restored image is written
    unless ``write_image`` is False, mimicking a run that exited 0 without output.
    """

    def __init__(self, returncode: int = 0, lines=None, write_image: bool = True):
        self.returncode = returncode
        self.lines = lines or ["WSClean version 3.7", "Cleaning up temporary files..."]
        self.write_image = write_image
        self.calls: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": list(cmd), **kwargs})
        outdir = Path(cmd[cmd.index("-name") + 1]).parent
        if self.returncode == 0 and self.write_image:
            (outdir / imaging_products.IMAGE_FILENAME).write_bytes(b"SIMPLE")
        lines = self.lines
        rc = self.returncode

        class _Proc:
            stdout = iter(line + "\n" for line in lines)

            @staticmethod
            def wait():
                return rc

            @staticmethod
            def kill():
                pass

        return _Proc()


def _wsclean_cmd(outdir: Path) -> list[str]:
    return ["wsclean", "-name", str(outdir / "wsclean"), "-spws", "1", "in.ms"]


@pytest.mark.unit
def test_imaging_marker_paths_sit_next_to_the_spw_directory(tmp_path):
    outdir = tmp_path / "ms" / "SPW-3"
    assert imaging_markers.imaging_marker_path(outdir) == tmp_path / "ms" / "SPW-3.done"
    assert imaging_markers.imaging_failure_marker_path(outdir) == tmp_path / "ms" / "SPW-3.failed"
    assert imaging_markers.imaging_log_path(outdir) == tmp_path / "ms" / "SPW-3.log"
    assert not imaging_markers.is_imaging_complete(outdir)
    outdir.mkdir(parents=True)
    imaging_markers.write_imaging_marker(outdir, ms_path="in.ms", spw=3)
    assert not imaging_markers.is_imaging_complete(outdir), (
        "a marker without the image is not complete"
    )
    (outdir / imaging_products.IMAGE_FILENAME).write_bytes(b"")
    assert imaging_markers.is_imaging_complete(outdir)


@pytest.mark.unit
def test_wsclean_environment_pins_openblas_to_one_thread():
    env = runtime_env.wsclean_environment(6, base={"LD_LIBRARY_PATH": "/spack/lib", "PATH": "/bin"})
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "6"
    assert env["LD_LIBRARY_PATH"].startswith("/lib64:")
    assert env["LD_LIBRARY_PATH"].endswith(":/spack/lib")
    assert env["PATH"] == "/bin"


@pytest.mark.unit
def test_run_wsclean_task_success_writes_done_marker_and_log(tmp_path, monkeypatch):
    fake = _FakeWsclean(returncode=0)
    monkeypatch.setattr(subprocess, "Popen", fake)
    outdir = tmp_path / "ms" / "SPW-1"

    result = imaging_task.run_wsclean_task(
        command=_wsclean_cmd(outdir), output_dir=str(outdir), ms_path="in.ms", spw=1, threads=4
    )

    assert imaging_markers.is_imaging_complete(outdir)
    assert result["image"] == str(outdir / imaging_products.IMAGE_FILENAME)
    payload = json.loads(imaging_markers.imaging_marker_path(outdir).read_text())
    assert payload["spw"] == 1 and payload["ms"] == "in.ms" and "wsclean" in payload["command"]
    log_text = imaging_markers.imaging_log_path(outdir).read_text()
    assert log_text.splitlines()[0].startswith("# wsclean -name")
    assert "Cleaning up temporary files" in log_text
    assert fake.calls[0]["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert fake.calls[0]["env"]["OMP_NUM_THREADS"] == "4"
    assert not imaging_markers.imaging_failure_marker_path(outdir).exists()


@pytest.mark.unit
def test_run_wsclean_task_hard_abort_writes_failure_marker(tmp_path, monkeypatch):
    lines = [
        "Gridding...",
        "terminate called after throwing an instance of 'std::bad_alloc'",
        "  what():  std::bad_alloc",
    ]
    monkeypatch.setattr(subprocess, "Popen", _FakeWsclean(returncode=-6, lines=lines))
    outdir = tmp_path / "ms" / "SPW-1"

    with pytest.raises(RuntimeError, match="return code -6"):
        imaging_task.run_wsclean_task(
            command=_wsclean_cmd(outdir), output_dir=str(outdir), ms_path="in.ms", spw=1, threads=2
        )

    marker = imaging_markers.imaging_failure_marker_path(outdir)
    assert marker.is_file()
    payload = json.loads(marker.read_text())
    assert payload["returncode"] == -6
    assert "what():  std::bad_alloc" in payload["error"]
    assert Path(payload["log"]).is_file()
    assert payload["stage"] == "image"
    assert not imaging_markers.imaging_marker_path(outdir).exists()


@pytest.mark.unit
def test_run_wsclean_task_removes_stale_files_of_a_previous_attempt(tmp_path, monkeypatch):
    fake = _FakeWsclean(returncode=0)
    monkeypatch.setattr(subprocess, "Popen", fake)
    outdir = tmp_path / "ms" / "SPW-1"
    outdir.mkdir(parents=True)
    # What a crashed or killed attempt leaves behind: WSClean's reorder files
    # and the single-window copy (hundreds of GB across a production run).
    (outdir / "in.ms-part0000-I.tmp").write_bytes(b"x")
    (outdir / "in.ms-part0000-I-w.tmp").write_bytes(b"x")
    (outdir / "spw1.single.ms").mkdir()
    (outdir / "spw1.single.ms" / "table.dat").write_bytes(b"x")
    imaging_markers.write_imaging_failure_marker(outdir, ms_path="in.ms", spw=1, error="old")

    imaging_task.run_wsclean_task(
        command=_wsclean_cmd(outdir), output_dir=str(outdir), ms_path="in.ms", spw=1, threads=2
    )

    assert not list(outdir.glob("*.tmp"))
    assert not (outdir / "spw1.single.ms").exists()
    assert imaging_markers.is_imaging_complete(outdir)
    assert not imaging_markers.imaging_failure_marker_path(outdir).exists()


@pytest.mark.unit
def test_run_wsclean_task_without_image_is_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _FakeWsclean(returncode=0, write_image=False))
    outdir = tmp_path / "ms" / "SPW-2"

    with pytest.raises(RuntimeError, match="wrote no wsclean-image.fits"):
        imaging_task.run_wsclean_task(
            command=_wsclean_cmd(outdir), output_dir=str(outdir), ms_path="in.ms", spw=2, threads=2
        )
    assert imaging_markers.imaging_failure_marker_path(outdir).is_file()


@pytest.mark.unit
def test_run_wsclean_task_missing_binary_is_a_failure(tmp_path, monkeypatch):
    def _no_binary(*args, **kwargs):
        raise FileNotFoundError("wsclean")

    monkeypatch.setattr(subprocess, "Popen", _no_binary)
    outdir = tmp_path / "ms" / "SPW-2"
    with pytest.raises(RuntimeError, match="Could not start wsclean"):
        imaging_task.run_wsclean_task(
            command=_wsclean_cmd(outdir), output_dir=str(outdir), ms_path="in.ms", spw=2, threads=2
        )
    payload = json.loads(imaging_markers.imaging_failure_marker_path(outdir).read_text())
    assert "Could not start wsclean" in payload["error"]


@pytest.mark.unit
def test_run_wsclean_task_replaces_stale_markers(tmp_path, monkeypatch):
    outdir = tmp_path / "ms" / "SPW-1"
    outdir.mkdir(parents=True)
    imaging_markers.write_imaging_failure_marker(outdir, ms_path="in.ms", spw=1, error="old")
    monkeypatch.setattr(subprocess, "Popen", _FakeWsclean(returncode=0))

    imaging_task.run_wsclean_task(
        command=_wsclean_cmd(outdir), output_dir=str(outdir), ms_path="in.ms", spw=1, threads=1
    )
    assert imaging_markers.is_imaging_complete(outdir)
    assert not imaging_markers.imaging_failure_marker_path(outdir).exists()


@pytest.mark.unit
def test_build_imaging_tasks_skips_done_and_grandfathers_images(tmp_path):
    csv_path = _params_csv(tmp_path, rows=3)
    parameters = pd.read_csv(csv_path)
    out = tmp_path / "images"
    done_dir = out / "uid___A001_X1_X1.ms.split" / "SPW-0"
    done_dir.mkdir(parents=True)
    (done_dir / imaging_products.IMAGE_FILENAME).write_bytes(b"")
    imaging_markers.write_imaging_marker(done_dir, ms_path="x", spw=0)
    legacy_dir = out / "uid___A001_X1_X1.ms.split" / "SPW-1"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / imaging_products.IMAGE_FILENAME).write_bytes(b"")

    common = dict(fov_fraction=1.5, beam_sampling=8, num_cores=10, max_cores_per_node=95)

    tasks, skipped = cli_image.build_imaging_tasks(parameters, out, **common)
    assert skipped == 1
    assert [t.spw for t in tasks] == [1, 2], "an image without a marker is retried by default"

    tasks, skipped = cli_image.build_imaging_tasks(
        parameters, out, trust_existing_images=True, **common
    )
    assert skipped == 2
    assert [t.spw for t in tasks] == [2]
    assert imaging_markers.imaging_marker_path(legacy_dir).is_file(), (
        "grandfathered image gets its marker"
    )

    tasks, skipped = cli_image.build_imaging_tasks(
        parameters, out, overwrite_outputs=True, **common
    )
    assert skipped == 0 and [t.spw for t in tasks] == [0, 1, 2]
    assert "-field" not in tasks[0].command, "no target_field_ids column: no field selection"


class _FakeFuture:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.cancelled = False

    def done(self):
        return True

    def exception(self):
        return self._error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self):
        self.cancelled = True


class _FakeBackend:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.submitted: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def submit_callable(self, func, *, cores, **kwargs):
        self.submitted.append({"func": func, "cores": cores, **kwargs})
        outcome = self.outcomes[len(self.submitted) - 1]
        if isinstance(outcome, Exception):
            return _FakeFuture(error=outcome)
        return _FakeFuture(result=outcome)


def _tasks(tmp_path: Path, n: int) -> list[cli_image.ImagingTask]:
    out = tmp_path / "images" / "ms"
    return [
        cli_image.ImagingTask(
            label=f"ms_{i}",
            ms_path=Path("in.ms"),
            spw=i,
            output_dir=out / f"SPW-{i}",
            command=["wsclean", "-name", str(out / f"SPW-{i}" / "wsclean"), "in.ms"],
        )
        for i in range(n)
    ]


def _run(tasks, backend, monkeypatch, **overrides):
    monkeypatch.setattr("almasim.services.compute.create_backend", lambda *a, **k: backend)
    monkeypatch.setattr(batch_runner, "tqdm", _QuietTqdm)
    from almasim.services.imaging.options import ExecOptions, ImagingOptions

    execution = dict(
        backend=overrides.pop("backend_kind", "slurm"),
        cores_per_task=4,
        node_cores=32,
        queue="q",
        project=None,
        walltime="01:00:00",
        memory="8GB",
        n_jobs=2,
        task_timeout=10.0,
    )
    for name in ("cores_per_task", "node_cores", "n_jobs", "continue_on_error"):
        if name in overrides:
            execution[name] = overrides.pop(name)
    imaging = ImagingOptions(pbcor=overrides.pop("pbcor", False))
    return batch_runner.run_imaging_tasks(
        tasks, imaging, ExecOptions(**execution), heartbeat_interval=1e9, **overrides
    )


class _QuietTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def update(self, n=1):
        pass

    def write(self, text):
        pass

    def set_postfix_str(self, text):
        pass


@pytest.mark.unit
def test_run_imaging_tasks_continues_past_failures_and_marks_lost_tasks(tmp_path, monkeypatch):
    tasks = _tasks(tmp_path, 3)
    backend = _FakeBackend(
        [{"ok": 0}, RuntimeError("WSClean exited with return code 1"), {"ok": 2}]
    )
    failures: list[cli_image.TaskFailure] = []

    results = _run(tasks, backend, monkeypatch, failures=failures)

    assert results == [{"ok": 0}, {"ok": 2}]
    assert [f.label for f in failures] == ["ms_1"]
    assert "return code 1" in failures[0].error
    # the worker never wrote a marker for the lost task: the driver does it
    payload = json.loads(
        imaging_markers.imaging_failure_marker_path(tasks[1].output_dir).read_text()
    )
    assert payload["spw"] == 1 and "return code 1" in payload["error"]
    assert not imaging_markers.imaging_failure_marker_path(tasks[0].output_dir).exists()
    # resources and task kwargs both reach the backend
    assert backend.submitted[0]["cores"] == 4
    assert backend.submitted[0]["threads"] == 4
    assert backend.submitted[0]["func"] is imaging_task.run_wsclean_task
    assert backend.submitted[0]["timeout"] == 10.0


class _SlowFuture(_FakeFuture):
    """Finishes on its second ``done()`` poll, so tasks overlap in flight."""

    def __init__(self, backend, result):
        super().__init__(result=result)
        self._backend = backend
        self._polls = 0

    def done(self):
        self._polls += 1
        if self._polls < 2:
            return False
        self._backend.finished += 1
        return True


class _SlowBackend(_FakeBackend):
    def __init__(self):
        super().__init__([])
        self.finished = 0
        self.max_in_flight = 0
        self.waited_for = None

    def wait_for_workers(self, n_workers, timeout=None):
        self.waited_for = (n_workers, timeout)
        return n_workers

    def submit_callable(self, func, *, cores, **kwargs):
        self.submitted.append({"func": func, "cores": cores, **kwargs})
        in_flight = len(self.submitted) - self.finished
        self.max_in_flight = max(self.max_in_flight, in_flight)
        return _SlowFuture(self, {"spw": kwargs["spw"]})


@pytest.mark.unit
def test_run_imaging_tasks_waits_for_workers_and_fills_only_free_slots(tmp_path, monkeypatch):
    """Job 3879: all 46 tasks landed on the first worker to register because
    they were all submitted at once; Dask never moves them afterwards."""
    monkeypatch.setattr(batch_runner, "sleep", lambda s: None)
    tasks = _tasks(tmp_path, 11)
    backend = _SlowBackend()

    # 2 workers x (32 // 12) = 4 slots
    results = _run(tasks, backend, monkeypatch, cores_per_task=12, node_cores=32, n_jobs=2)

    assert backend.waited_for[0] == 2
    assert backend.max_in_flight == 4
    assert sorted(r["spw"] for r in results) == list(range(11))


@pytest.mark.unit
def test_run_imaging_tasks_fail_fast_stops_and_cancels(tmp_path, monkeypatch):
    tasks = _tasks(tmp_path, 2)
    backend = _FakeBackend([RuntimeError("boom"), {"ok": 1}])
    failures: list[cli_image.TaskFailure] = []

    with pytest.raises(batch_runner.ImagingRunAborted, match="ms_0"):
        _run(tasks, backend, monkeypatch, failures=failures, continue_on_error=False)
    assert [f.label for f in failures] == ["ms_0"]


@pytest.mark.unit
def test_run_imaging_tasks_sync_backend_runs_in_process(tmp_path, monkeypatch):
    tasks = _tasks(tmp_path, 2)
    calls = []

    def _fake_task(**kwargs):
        calls.append(kwargs)
        if kwargs["spw"] == 0:
            raise RuntimeError("no image")
        return {"spw": kwargs["spw"]}

    patch_everywhere(monkeypatch, imaging_task, "run_wsclean_task", _fake_task)
    failures: list[cli_image.TaskFailure] = []
    results = _run(tasks, None, monkeypatch, backend_kind="sync", failures=failures)

    assert results == [{"spw": 1}]
    assert [f.label for f in failures] == ["ms_0"]
    assert calls[0]["threads"] == 4
    assert imaging_markers.imaging_failure_marker_path(tasks[0].output_dir).is_file()


@pytest.mark.unit
def test_image_from_ms_reports_failures_and_exits_one(monkeypatch, tmp_path):
    csv_path = _params_csv(tmp_path, rows=2)

    def _fake_run(tasks, imaging, execution, *, failures, **kwargs):
        failures.append(
            cli_image.TaskFailure(tasks[0].label, "WSClean exited with return code 1", "/l")
        )
        return [{"ok": 1}]

    monkeypatch.setattr(cli_image, "run_imaging_tasks", _fake_run)
    result = runner.invoke(
        cli.app,
        [
            "image",
            "image-from-ms",
            str(csv_path),
            str(tmp_path / "out"),
            "--postprocess-backend",
            "sync",
        ],
    )
    assert result.exit_code == 1
    assert "Imaged 1/2 task(s), 1 failed." in result.output
    assert "1 task(s) failed and were skipped" in result.output
    assert "log: /l" in result.output


@pytest.mark.unit
def test_image_from_ms_nothing_to_do_when_all_done(monkeypatch, tmp_path):
    csv_path = _params_csv(tmp_path, rows=1)
    out = tmp_path / "out"
    done_dir = out / "uid___A001_X1_X1.ms.split" / "SPW-0"
    done_dir.mkdir(parents=True)
    (done_dir / imaging_products.IMAGE_FILENAME).write_bytes(b"")
    imaging_markers.write_imaging_marker(done_dir, ms_path="x", spw=0)
    monkeypatch.setattr(cli_image, "run_imaging_tasks", lambda *a, **k: pytest.fail("no run"))

    result = runner.invoke(cli.app, ["image", "image-from-ms", str(csv_path), str(out)])
    assert result.exit_code == 0, result.output
    assert "Skipped 1 already-imaged SPW(s)." in result.output
    assert "nothing to do" in result.output


@pytest.mark.unit
def test_image_from_ms_rejects_unknown_backend(tmp_path):
    csv_path = _params_csv(tmp_path, rows=1)
    result = runner.invoke(
        cli.app,
        ["image", "image-from-ms", str(csv_path), str(tmp_path), "--postprocess-backend", "k8s"],
    )
    assert result.exit_code == 2


@pytest.mark.unit
def test_compute_parameters_records_unreadable_ms_and_continues(monkeypatch, tmp_path):
    folder = tmp_path / "cal"
    folder.mkdir()
    for name in ("a.ms.split.cal", "b.ms.split.cal", "c.ms.split.cal"):
        (folder / name).mkdir()
        if name != "c.ms.split.cal":
            (folder / f"{name}.done").write_text("{}")
    patch_everywhere(monkeypatch, cli_image, "tqdm", lambda iterable: iterable)

    def _fake_compute(ms: Path, science_only: bool = True, intents=None):
        if ms.name.startswith("b"):
            raise RuntimeError("Table does not exist")
        return pd.DataFrame({"filename": [str(ms)], "spectral_window_id": [0]})

    monkeypatch.setattr(cli_image, "compute_imaging_parameters", _fake_compute)
    out_csv = tmp_path / "params.csv"

    result = runner.invoke(cli.app, ["image", "compute-parameters", str(folder), str(out_csv)])

    assert result.exit_code == 1
    assert "Skipping 1 MS(s) without a .done calibration marker." in result.output
    assert "FAILED b.ms.split.cal" in result.output
    written = pd.read_csv(out_csv)
    assert list(written["filename"]) == [str(folder / "a.ms.split.cal")]
    failed = (tmp_path / "params.csv.failed.tsv").read_text()
    assert "b.ms.split.cal\tRuntimeError: Table does not exist" in failed


@pytest.mark.unit
def test_science_selection_reports_used_spws_and_target_fields(monkeypatch):
    class _Tab:
        def __init__(self, cols, nrows=None):
            self.cols = cols
            self._n = nrows

        def getcol(self, name):
            return self.cols[name]

        def nrows(self):
            return self._n if self._n is not None else len(next(iter(self.cols.values())))

    tables = {
        "::DATA_DESCRIPTION": _Tab({"SPECTRAL_WINDOW_ID": np.array([0, 5, 7])}),
        "::STATE": _Tab({"OBS_MODE": ["CALIBRATE_PHASE#ON_SOURCE", "OBSERVE_TARGET#ON_SOURCE"]}),
        "": _Tab(
            {
                "DATA_DESC_ID": np.array([1, 1, 2, 2, 2, 1]),
                "STATE_ID": np.array([0, 1, 1, 0, 1, 1]),
                "FIELD_ID": np.array([0, 3, 3, 0, 4, 3]),
            }
        ),
    }

    def _fake_table(name, ack=False):
        for suffix, tab in tables.items():
            if suffix and name.endswith(suffix):
                return tab
        return tables[""]

    patch_everywhere(monkeypatch, cli_image, "import_casacore_tables", lambda: _fake_table)
    rows_per_spw, target_fields = cli_image.science_selection(Path("x.ms"))
    assert rows_per_spw == {5: 3, 7: 3}, "SPW 0 has no rows and is left out"
    assert target_fields == [3, 4]


@pytest.mark.unit
def test_imaging_parameter_to_command_arg_threshold_flags():
    row = pd.Series(
        {"spectral_window_id": 1, "fov_per_frequency": 10.0, "synthetized_beam_size": 1.0}
    )
    plain = cli_image.imaging_parameter_to_command_arg(row, 1.0, 4.0)
    assert "-auto-threshold" not in plain and "-auto-mask" not in plain
    flagged = cli_image.imaging_parameter_to_command_arg(
        row, 1.0, 4.0, auto_threshold=2.5, auto_mask=5.0
    )
    assert flagged[flagged.index("-auto-threshold") + 1] == "2.5"
    assert flagged[flagged.index("-auto-mask") + 1] == "5.0"
    assert cli_image.imaging_parameter_to_command_arg(row, 1.0, 4.0, auto_threshold=0) == plain


@pytest.mark.unit
def test_image_pixels_min_beams_across_only_widens_narrow_images():
    row = pd.Series({"fov_per_frequency": 30.0, "synthetized_beam_size": 3.0})
    assert cli_image.image_pixels(row, 1.5, 8) == 120, "15 beams from the FOV"
    assert cli_image.image_pixels(row, 1.5, 8, min_beams_across=32) == 256
    assert cli_image.image_pixels(row, 1.5, 8, min_beams_across=10) == 120


@pytest.mark.unit
def test_imaging_parameter_to_command_arg_multiscale_and_min_beams():
    row = pd.Series(
        {"spectral_window_id": 1, "fov_per_frequency": 10.0, "synthetized_beam_size": 1.0}
    )
    plain = cli_image.imaging_parameter_to_command_arg(row, 1.0, 4.0)
    assert "-multiscale" not in plain
    assert plain[plain.index("-size") + 1] == "40"
    wide = cli_image.imaging_parameter_to_command_arg(
        row, 1.0, 4.0, min_beams_across=32, multiscale=True
    )
    assert "-multiscale" in wide
    assert wide[wide.index("-size") + 1 : wide.index("-size") + 3] == ["128", "128"]


@pytest.mark.unit
def test_build_imaging_tasks_abs_mem_and_scratch_placeholder(tmp_path):
    parameters = pd.read_csv(_params_csv(tmp_path, rows=1))
    tasks, _ = cli_image.build_imaging_tasks(
        parameters,
        tmp_path / "out",
        fov_fraction=1.5,
        beam_sampling=8,
        num_cores=10,
        max_cores_per_node=96,
        task_memory_gb=40,
        scratch_dir="/tmp",
    )
    cmd = tasks[0].command
    assert cmd[cmd.index("-abs-mem") + 1] == "40" and "-mem" not in cmd
    assert cmd[cmd.index("-temp-dir") + 1] == "__SCRATCH__"


@pytest.mark.unit
def test_run_wsclean_task_uses_and_removes_scratch_dir(tmp_path, monkeypatch):
    fake = _FakeWsclean(returncode=0)
    monkeypatch.setattr(subprocess, "Popen", fake)
    outdir = tmp_path / "ms" / "SPW-1"
    scratch_root = tmp_path / "scratch"
    cmd = ["wsclean", "-name", str(outdir / "wsclean"), "-temp-dir", "__SCRATCH__", "in.ms"]

    imaging_task.run_wsclean_task(
        command=cmd,
        output_dir=str(outdir),
        ms_path="in.ms",
        spw=1,
        threads=2,
        scratch_root=str(scratch_root),
    )
    used = fake.calls[0]["cmd"]
    temp = Path(used[used.index("-temp-dir") + 1])
    assert temp.parent == scratch_root and temp.name.startswith("SPW-1-")
    assert not temp.exists(), "per-task scratch is removed after the run"
    assert imaging_markers.is_imaging_complete(outdir)


@pytest.mark.unit
def test_run_wsclean_task_removes_scratch_dir_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _FakeWsclean(returncode=-11))
    outdir = tmp_path / "ms" / "SPW-1"
    scratch_root = tmp_path / "scratch"
    cmd = ["wsclean", "-name", str(outdir / "wsclean"), "-temp-dir", "__SCRATCH__", "in.ms"]
    with pytest.raises(RuntimeError):
        imaging_task.run_wsclean_task(
            command=cmd,
            output_dir=str(outdir),
            ms_path="in.ms",
            spw=1,
            threads=2,
            scratch_root=str(scratch_root),
        )
    assert list(scratch_root.iterdir()) == []
    assert imaging_markers.imaging_failure_marker_path(outdir).is_file()


class _FlakyWsclean:
    """Crashes with a signal ``crashes`` times, then behaves like ``final``."""

    def __init__(self, crashes: int, final: int = 0):
        self.remaining = crashes
        self.final = final
        self.calls = 0
        self.envs: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls += 1
        self.envs.append(kwargs)
        outdir = Path(cmd[cmd.index("-name") + 1]).parent
        if self.remaining > 0:
            self.remaining -= 1
            (outdir / "x-part0000-I.tmp").write_bytes(b"junk")
            return _FakeWsclean(returncode=-11, lines=["Reordering: 0%....10%"])(cmd, **kwargs)
        return _FakeWsclean(returncode=self.final)(cmd, **kwargs)


@pytest.mark.unit
def test_run_wsclean_task_retries_signal_crashes(tmp_path, monkeypatch):
    flaky = _FlakyWsclean(crashes=2)
    monkeypatch.setattr(subprocess, "Popen", flaky)
    outdir = tmp_path / "ms" / "SPW-1"

    result = imaging_task.run_wsclean_task(
        command=_wsclean_cmd(outdir),
        output_dir=str(outdir),
        ms_path="in.ms",
        spw=1,
        threads=2,
        retries=3,
    )

    assert flaky.calls == 3 and result["attempts"] == 3
    assert imaging_markers.is_imaging_complete(outdir)
    pads = [call["env"]["ALMASIM_WSCLEAN_PAD"] for call in flaky.envs]
    assert pads == ["", "x", "xx"], "each attempt shifts the environment layout"
    log = imaging_markers.imaging_log_path(outdir).read_text()
    assert "# attempt 1 died with signal 11; retrying (3 left)" in log
    assert "# attempt 2 died with signal 11; retrying (2 left)" in log
    assert not list(outdir.glob("*.tmp")), "reorder leftovers are removed between attempts"


@pytest.mark.unit
def test_run_wsclean_task_gives_up_after_retries(tmp_path, monkeypatch):
    flaky = _FlakyWsclean(crashes=10)
    monkeypatch.setattr(subprocess, "Popen", flaky)
    outdir = tmp_path / "ms" / "SPW-1"
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        imaging_task.run_wsclean_task(
            command=_wsclean_cmd(outdir),
            output_dir=str(outdir),
            ms_path="in.ms",
            spw=1,
            threads=2,
            retries=2,
        )
    assert flaky.calls == 3
    payload = json.loads(imaging_markers.imaging_failure_marker_path(outdir).read_text())
    assert payload["returncode"] == -11 and "after 3 attempts" in payload["error"]


@pytest.mark.unit
def test_run_wsclean_task_does_not_retry_error_exits(tmp_path, monkeypatch):
    fake = _FakeWsclean(returncode=255, lines=["+ >>> Error opening meta file"])
    monkeypatch.setattr(subprocess, "Popen", fake)
    outdir = tmp_path / "ms" / "SPW-1"
    with pytest.raises(RuntimeError, match="return code 255"):
        imaging_task.run_wsclean_task(
            command=_wsclean_cmd(outdir),
            output_dir=str(outdir),
            ms_path="in.ms",
            spw=1,
            threads=2,
            retries=3,
        )
    assert len(fake.calls) == 1


@pytest.mark.unit
def test_build_imaging_tasks_single_window_default(tmp_path):
    csv_path = tmp_path / "params.csv"
    pd.DataFrame(
        {
            "filename": ["/data/uid___A001_X1_X1.ms.split.cal"],
            "spectral_window_id": [5],
            "reference_frequency": [100.0e9],
            "fov_per_frequency": [8.0],
            "max_baseline_size": [100.0],
            "synthetized_beam_size": [2.0],
            "target_field_ids": ["2,3"],
        }
    ).to_csv(csv_path, index=False)
    tasks, _ = cli_image.build_imaging_tasks(
        pd.read_csv(csv_path),
        tmp_path / "out",
        fov_fraction=1.5,
        beam_sampling=8,
        num_cores=10,
        max_cores_per_node=96,
    )
    cmd = tasks[0].command
    assert cmd[-1] == imaging_products.SINGLE_WINDOW_PLACEHOLDER
    assert "-no-reorder" in cmd and "-spws" not in cmd
    assert cmd[cmd.index("-field") + 1] == "2,3", "WSClean images field 0 unless told otherwise"
    assert tasks[0].field_ids == (2, 3) and tasks[0].spw == 5


@pytest.mark.unit
def test_run_wsclean_task_single_window_extracts_and_cleans_up(tmp_path, monkeypatch):
    fake = _FakeWsclean(returncode=0)
    monkeypatch.setattr(subprocess, "Popen", fake)
    calls = []

    def fake_extract(
        ms_path, spw, field_ids, out_path, *, timeout=None, log_path=None, max_weight=None
    ):
        calls.append((ms_path, spw, list(field_ids), Path(out_path), max_weight))
        Path(out_path).mkdir(parents=True)
        return {
            "rows": 10,
            "bytes": 1,
            "data_desc_ids": [spw],
            "seconds": 0.1,
            "dropped_rows": 4,
            "max_weight": max_weight,
        }

    patch_everywhere(
        monkeypatch, imaging_ms_single_window, "extract_single_window_ms_subprocess", fake_extract
    )
    outdir = tmp_path / "ms" / "SPW-5"
    cmd = [
        "wsclean",
        "-name",
        str(outdir / "wsclean"),
        "-no-reorder",
        imaging_products.SINGLE_WINDOW_PLACEHOLDER,
    ]

    imaging_task.run_wsclean_task(
        command=cmd,
        output_dir=str(outdir),
        ms_path="/data/in.ms",
        spw=5,
        threads=2,
        single_window=True,
        field_ids=[2, 3],
    )
    assert calls == [
        (
            "/data/in.ms",
            5,
            [2, 3],
            outdir / "spw5.single.ms",
            imaging_ms_single_window.DEFAULT_MAX_WEIGHT,
        )
    ]
    used = fake.calls[0]["cmd"]
    assert used[-1] == str(outdir / "spw5.single.ms")
    assert not (outdir / "spw5.single.ms").exists(), "the per-task MS is removed afterwards"
    assert imaging_markers.is_imaging_complete(outdir)
    log_lines = imaging_markers.imaging_log_path(outdir).read_text().splitlines()
    assert log_lines[0].startswith("# extracted spectral window 5 fields [2, 3]")
    assert "dropped 4 rows (28.6 %) with WEIGHT > 10000" in log_lines[0]
    assert log_lines[1].startswith("# wsclean -name")


@pytest.mark.unit
def test_run_wsclean_task_single_window_extraction_failure_is_marked(tmp_path, monkeypatch):
    def failing_extract(*args, **kwargs):
        raise RuntimeError("No visibilities for spectral window 5")

    patch_everywhere(
        monkeypatch,
        imaging_ms_single_window,
        "extract_single_window_ms_subprocess",
        failing_extract,
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("WSClean must not run"))
    outdir = tmp_path / "ms" / "SPW-5"
    with pytest.raises(RuntimeError, match="Could not extract spectral window 5"):
        imaging_task.run_wsclean_task(
            command=[
                "wsclean",
                "-name",
                str(outdir / "wsclean"),
                imaging_products.SINGLE_WINDOW_PLACEHOLDER,
            ],
            output_dir=str(outdir),
            ms_path="/data/in.ms",
            spw=5,
            threads=2,
            single_window=True,
        )
    payload = json.loads(imaging_markers.imaging_failure_marker_path(outdir).read_text())
    assert "No visibilities" in payload["error"]


class _Child:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


_LOCK_FILE_STDERR = (
    "RuntimeError: (/code/casa/IO/LockFile.cc : 300) Failed AlwaysAssert tracePWRITE (...)\n"
)


@pytest.mark.unit
def test_extract_single_window_ms_retries_lock_file_errors_in_a_new_process(tmp_path, monkeypatch):
    """casacore keeps a failed copy's output in its table cache, so a retry in
    the same process fails with "target table ... is still open" (job 3841);
    every attempt must be a new child."""
    children = []

    def fake_run(cmd, **kwargs):
        children.append(cmd)
        if len(children) < 3:
            return _Child(1, stderr=_LOCK_FILE_STDERR)
        return _Child(0, stdout='{"rows": 1, "bytes": 1, "data_desc_ids": [5]}')

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    log_path = tmp_path / "SPW-5.log"
    result = imaging_ms_single_window.extract_single_window_ms_subprocess(
        "in.ms", 5, [2], tmp_path / "out.ms", log_path=log_path, attempts=5, retry_seconds=0
    )
    assert len(children) == 3 and result["attempts"] == 3
    assert log_path.read_text().count("retrying in a new process") == 2


@pytest.mark.unit
def test_extract_single_window_ms_gives_up_after_the_last_lock_file_attempt(tmp_path, monkeypatch):
    children = []

    def fake_run(cmd, **kwargs):
        children.append(cmd)
        return _Child(1, stderr=_LOCK_FILE_STDERR)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="LockFile.cc"):
        imaging_ms_single_window.extract_single_window_ms_subprocess(
            "in.ms", 5, [2], tmp_path / "out.ms", attempts=3, retry_seconds=0
        )
    assert len(children) == 3


@pytest.mark.unit
def test_extract_single_window_ms_does_not_retry_other_errors(tmp_path, monkeypatch):
    children = []

    def fake_run(cmd, **kwargs):
        children.append(cmd)
        return _Child(1, stderr="RuntimeError: No visibilities for spectral window 5\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="No visibilities"):
        imaging_ms_single_window.extract_single_window_ms_subprocess(
            "in.ms", 5, [2], tmp_path / "out.ms", attempts=5, retry_seconds=0
        )
    assert len(children) == 1


@pytest.mark.unit
def test_extract_single_window_ms_runs_in_a_child_process(tmp_path):
    """The copy must not run in the worker: casacore holds the GIL and the
    Dask scheduler drops a worker whose event loop stops answering."""
    pytest.importorskip("casacore.tables")
    log_path = tmp_path / "SPW-5.log"
    with pytest.raises(RuntimeError, match="extraction exited with return code"):
        imaging_ms_single_window.extract_single_window_ms_subprocess(
            tmp_path / "missing.ms", 5, [2], tmp_path / "spw5.single.ms", log_path=log_path
        )
    assert "missing.ms" in log_path.read_text(), "the child's traceback lands in the task log"


@pytest.mark.unit
def test_extract_single_window_ms_subprocess_reports_the_child_result(tmp_path, monkeypatch):
    class _Done:
        returncode = 0
        stdout = '{"rows": 12, "bytes": 3000, "data_desc_ids": [1]}'
        stderr = ""

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen.update(kwargs)
        return _Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = imaging_ms_single_window.extract_single_window_ms_subprocess(
        "/data/in.ms", 5, [2, 3], tmp_path / "out.ms", timeout=30, max_weight=500.0
    )
    assert seen["cmd"][0] == sys.executable and seen["cmd"][1] == "-c"
    assert json.loads(seen["cmd"][3]) == {
        "target": "almasim.services.imaging.ms.single_window:extract_single_window_ms",
        "kwargs": {
            "ms_path": "/data/in.ms",
            "spw": 5,
            "field_ids": [2, 3],
            "out_path": str(tmp_path / "out.ms"),
            "max_weight": 500.0,
        },
    }
    assert seen["timeout"] == 30 and seen["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert result["rows"] == 12 and "seconds" in result


@pytest.mark.unit
def test_build_imaging_tasks_makes_relative_paths_absolute(tmp_path, monkeypatch):
    """WSClean runs with the task directory as cwd, so relative paths broke every task.

    A run started from /SKAO_data/cycle11 with output ``ms-imaged-flagged`` had WSClean
    look for ``.../SPW-5/ms-imaged-flagged/.../spw5.single.ms`` (2026-10-09).
    """
    csv_path = _params_csv(tmp_path, rows=1)
    parameters = pd.read_csv(csv_path)
    monkeypatch.chdir(tmp_path)

    tasks, _ = cli_image.build_imaging_tasks(
        parameters,
        Path("images"),
        fov_fraction=1.5,
        beam_sampling=8,
        num_cores=10,
        max_cores_per_node=95,
    )

    task = tasks[0]
    assert Path(task.output_dir).is_absolute()
    assert Path(task.output_dir).is_relative_to(tmp_path / "images")
    for flag in ("-name", "-temp-dir"):
        value = task.command[task.command.index(flag) + 1]
        assert Path(value).is_absolute(), f"{flag} {value} is relative"


def _geometry_tables(names):
    tables = {
        "::SPECTRAL_WINDOW": FakeTable(
            {
                "REF_FREQUENCY": np.array([100.0e9, 200.0e9, 230.0e9]),
                "NAME": np.array(["WVR", "X1#BB_1#SW-01#FULL_RES", "X1#BB_2#SW-01#FULL_RES"]),
                "NUM_CHAN": np.array([128, 3840, 480]),
                "CHAN_WIDTH": [np.full(n, 1e6) for n in (128, 3840, 480)],
                "RESOLUTION": [np.full(n, 2e6) for n in (128, 3840, 480)],
                "TOTAL_BANDWIDTH": np.array([128e6, 3840e6, 480e6]),
            }
        ),
        "::ANTENNA": FakeTable(
            {
                "DISH_DIAMETER": np.array([12.0, 12.0]),
                "POSITION": np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
            }
        ),
        "::OBSERVATION": [{"TIME_RANGE": [57000 * 86400.0, 57000.1 * 86400.0]}],
        "::FIELD": FakeTable({"NAME": np.array(names)}),
    }
    return lambda name, ack=False: next(t for k, t in tables.items() if name.endswith(k))


# uid___A002_X10d9399_X32fd: HOPS-358 (field 2) and HOPS-373 (field 4), 12.4' apart,
# were imaged together with ``-field 2,4``. Fields 0, 1, 3 are calibrators.
_X32FD_NAMES = ["J0538-4405", "J0552+0313", "HOPS-358", "J0551+0829", "HOPS-373"]
_X32FD_SCAN = (
    {1: 140, 2: 75},
    [2, 4],
    {(1, 0): 10, (1, 1): 20, (1, 2): 50, (1, 3): 20, (1, 4): 40, (2, 1): 15, (2, 2): 60},
)


@pytest.mark.unit
def test_compute_imaging_parameters_one_row_per_target_field(monkeypatch):
    """Science-only: a row per window and target field, one id per row, no calibrators."""
    patch_everywhere(
        monkeypatch, cli_image, "import_casacore_tables", lambda: _geometry_tables(_X32FD_NAMES)
    )
    patch_everywhere(monkeypatch, cli_image, "_field_scan", lambda _ms, _intents=(): _X32FD_SCAN)

    out = cli_image.compute_imaging_parameters(Path("x.ms.split.cal"))

    rows = list(
        zip(
            out["spectral_window_id"],
            out["target_field_ids"],
            out["n_visibility_rows"],
            strict=True,
        )
    )
    assert rows == [(1, "2", 50), (1, "4", 40), (2, "2", 60)], "field 4 has no rows in SPW 2"
    assert list(out["field_name"]) == ["HOPS-358", "HOPS-373", "HOPS-358"]
    assert "field_id" not in out.columns, "the field id is written once, in target_field_ids"


@pytest.mark.unit
def test_compute_imaging_parameters_all_spws_includes_calibrators(monkeypatch):
    """--all-spws: every window, a row per field with data in it, calibrators included."""
    patch_everywhere(
        monkeypatch, cli_image, "import_casacore_tables", lambda: _geometry_tables(_X32FD_NAMES)
    )
    patch_everywhere(monkeypatch, cli_image, "_field_scan", lambda _ms, _intents=(): _X32FD_SCAN)

    out = cli_image.compute_imaging_parameters(Path("x.ms.split.cal"), science_only=False)

    rows = list(
        zip(out["spectral_window_id"], out["target_field_ids"], out["field_name"], strict=True)
    )
    assert rows == [  # SPW 0 has no visibilities: no row
        (1, "0", "J0538-4405"),
        (1, "1", "J0552+0313"),
        (1, "2", "HOPS-358"),
        (1, "3", "J0551+0829"),
        (1, "4", "HOPS-373"),
        (2, "1", "J0552+0313"),
        (2, "2", "HOPS-358"),
    ]


@pytest.mark.unit
def test_compute_imaging_parameters_without_selected_intent_gives_no_rows(monkeypatch):
    """No field with the intent: nothing to image (no fallback to WSClean's field 0)."""
    patch_everywhere(
        monkeypatch, cli_image, "import_casacore_tables", lambda: _geometry_tables(["a"])
    )
    patch_everywhere(
        monkeypatch,
        cli_image,
        "_field_scan",
        lambda _ms, _intents=(): ({0: 7, 2: 9}, [], {(0, 0): 7, (2, 0): 9}),
    )

    assert cli_image.compute_imaging_parameters(Path("x.ms")).empty


@pytest.mark.unit
def test_compute_imaging_parameters_passes_intents_and_rejects_them_with_all_fields(monkeypatch):
    seen = []

    def _scan(_ms, intents=()):
        seen.append(intents)
        return _X32FD_SCAN[0], [1], _X32FD_SCAN[2]  # field 1 = phase calibrator

    patch_everywhere(
        monkeypatch, cli_image, "import_casacore_tables", lambda: _geometry_tables(_X32FD_NAMES)
    )
    patch_everywhere(monkeypatch, cli_image, "_field_scan", _scan)

    out = cli_image.compute_imaging_parameters(Path("x.ms"), intents=["CALIBRATE_PHASE#ON_SOURCE"])
    assert seen == [("CALIBRATE_PHASE#ON_SOURCE",)]
    assert list(zip(out["spectral_window_id"], out["field_name"], strict=True)) == [
        (1, "J0552+0313"),
        (2, "J0552+0313"),
    ]
    with pytest.raises(ValueError):
        cli_image.compute_imaging_parameters(Path("x.ms"), science_only=False, intents=["X"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("obs_mode", "intents", "expected"),
    [
        ("CALIBRATE_PHASE#ON_SOURCE,CALIBRATE_WVR#ON_SOURCE", ["CALIBRATE_PHASE#ON_SOURCE"], True),
        ("CALIBRATE_PHASE#ON_SOURCE,CALIBRATE_WVR#ON_SOURCE", ["CALIBRATE_PHASE"], True),
        ("CALIBRATE_PHASE#OFF_SOURCE", ["CALIBRATE_PHASE#ON_SOURCE"], False),
        ("OBSERVE_TARGET#ON_SOURCE", ["OBSERVE_TARGET"], True),
        ("OBSERVE_TARGET#ON_SOURCE", ["OBSERVE"], False),
        ("CALIBRATE_BANDPASS#ON_SOURCE", ["CALIBRATE_FLUX", "CALIBRATE_BANDPASS"], True),
    ],
)
def test_intent_matches(obs_mode, intents, expected):
    assert intent_matches(obs_mode, intents) is expected


@pytest.mark.unit
def test_field_scan_selects_fields_by_given_intent(monkeypatch):
    class _Tab:
        def __init__(self, cols):
            self.cols = cols

        def getcol(self, name):
            return self.cols[name]

        def nrows(self):
            return len(next(iter(self.cols.values())))

    tables = {
        "::DATA_DESCRIPTION": _Tab({"SPECTRAL_WINDOW_ID": np.array([0])}),
        "::STATE": _Tab(
            {
                "OBS_MODE": [
                    "CALIBRATE_PHASE#ON_SOURCE,CALIBRATE_WVR#ON_SOURCE",
                    "OBSERVE_TARGET#ON_SOURCE",
                ]
            }
        ),
        "": _Tab(
            {
                "DATA_DESC_ID": np.array([0, 0, 0]),
                "STATE_ID": np.array([0, 1, 1]),
                "FIELD_ID": np.array([1, 2, 4]),
            }
        ),
    }
    patch_everywhere(
        monkeypatch,
        cli_image,
        "import_casacore_tables",
        lambda: (
            lambda name, ack=False: next(
                (t for k, t in tables.items() if k and name.endswith(k)), tables[""]
            )
        ),
    )
    assert cli_image._field_scan(Path("x.ms"))[1] == [2, 4]
    assert cli_image._field_scan(Path("x.ms"), ("CALIBRATE_PHASE#ON_SOURCE",))[1] == [1]


@pytest.mark.unit
def test_compute_parameters_cli_rejects_intent_with_all_spws(tmp_path):
    result = runner.invoke(
        cli.app,
        ["image", "compute-parameters", str(tmp_path), "--all-spws", "--intent", "CALIBRATE_PHASE"],
    )
    assert result.exit_code != 0


@pytest.mark.unit
def test_field_scan_counts_rows_per_window_and_field(monkeypatch):
    class _Tab:
        def __init__(self, cols):
            self.cols = cols

        def getcol(self, name):
            return self.cols[name]

        def nrows(self):
            return len(next(iter(self.cols.values())))

    tables = {
        "::DATA_DESCRIPTION": _Tab({"SPECTRAL_WINDOW_ID": np.array([0, 5, 7])}),
        "::STATE": _Tab(
            {
                "OBS_MODE": [
                    "CALIBRATE_PHASE#ON_SOURCE",
                    "OBSERVE_TARGET#ON_SOURCE",
                    "CALIBRATE_ATMOSPHERE#ON_SOURCE",
                ]
            }
        ),
        "": _Tab(
            {
                # dd 1 = spw 5, dd 2 = spw 7; field 0 a calibrator, 3 and 4 targets;
                # the last row is field 3 under the atmosphere intent and still counts.
                "DATA_DESC_ID": np.array([1, 1, 2, 2, 2, 1, 1]),
                "STATE_ID": np.array([0, 1, 1, 0, 1, 1, 2]),
                "FIELD_ID": np.array([0, 3, 3, 0, 4, 4, 3]),
            }
        ),
    }

    def _fake_table(name, ack=False):
        return next((t for k, t in tables.items() if k and name.endswith(k)), tables[""])

    patch_everywhere(monkeypatch, cli_image, "import_casacore_tables", lambda: _fake_table)
    rows_per_spw, targets, per_field = cli_image._field_scan(Path("x.ms"))
    assert rows_per_spw == {5: 4, 7: 3}
    assert targets == [3, 4]
    assert per_field == {(5, 0): 1, (5, 3): 2, (5, 4): 1, (7, 0): 1, (7, 3): 1, (7, 4): 1}


@pytest.mark.unit
def test_build_imaging_tasks_one_task_and_directory_per_field(tmp_path):
    common = dict(fov_fraction=1.5, beam_sampling=8, num_cores=10, max_cores_per_node=95)
    row = dict(
        filename=str(tmp_path / "uid___A002_X1_X2.ms.split.cal"),
        spectral_window_id=23,
        reference_frequency=230e9,
        fov_per_frequency=25.0,
        max_baseline_size=8500.0,
        synthetized_beam_size=0.03,
    )
    per_field = pd.DataFrame(
        [
            {**row, "target_field_ids": "2", "field_name": "HOPS-358"},
            {**row, "target_field_ids": "4", "field_name": "HOPS-373"},
        ]
    )
    csv_path = tmp_path / "params.csv"
    per_field.to_csv(csv_path, index=False)

    tasks, skipped = cli_image.build_imaging_tasks(
        pd.read_csv(csv_path), tmp_path / "img", **common
    )

    assert skipped == 0
    assert [t.output_dir.name for t in tasks] == ["SPW-23_FIELD-2", "SPW-23_FIELD-4"]
    assert [t.field_ids for t in tasks] == [(2,), (4,)]
    for task, field in zip(tasks, ("2", "4"), strict=True):
        assert task.command[task.command.index("-field") + 1] == field
    assert len({t.label for t in tasks}) == 2

    legacy = pd.DataFrame([{**row, "target_field_ids": "2,4"}])
    tasks, _ = cli_image.build_imaging_tasks(legacy, tmp_path / "img", **common)
    assert [t.output_dir.name for t in tasks] == ["SPW-23"], "old per-window CSVs keep their layout"


@pytest.mark.unit
@pytest.mark.parametrize(("pbcor", "expected"), [(True, 12.0), (False, None)])
def test_run_imaging_tasks_passes_the_dish_diameter_when_correcting(
    tmp_path, monkeypatch, pbcor, expected
):
    tasks = [dataclasses.replace(task, dish_diameter=12.0) for task in _tasks(tmp_path, 1)]
    backend = _FakeBackend([{"ok": 0}])
    _run(tasks, backend, monkeypatch, pbcor=pbcor)
    assert backend.submitted[0]["pb_dish_diameter"] == expected

    calls = []
    patch_everywhere(
        monkeypatch, imaging_task, "run_wsclean_task", lambda **kwargs: calls.append(kwargs) or {}
    )
    _run(tasks, None, monkeypatch, backend_kind="sync", pbcor=pbcor)
    assert calls[0]["pb_dish_diameter"] == expected


@pytest.mark.unit
def test_run_imaging_tasks_raises_when_the_workers_never_come_up(tmp_path, monkeypatch):
    class _NoWorkers(_FakeBackend):
        def wait_for_workers(self, n_workers, timeout=None):
            raise RuntimeError("no worker registered in 30 min")

    with pytest.raises(batch_runner.WorkersUnavailableError, match="no worker registered"):
        _run(_tasks(tmp_path, 1), _NoWorkers([]), monkeypatch)


@pytest.mark.unit
def test_ms_overview_fails_cleanly_without_casacore(monkeypatch, tmp_path):
    def _missing():
        raise cli_image.MissingDependencyError("Missing optional dependency `python-casacore`")

    patch_everywhere(monkeypatch, cli_image, "import_casacore_tables", _missing)
    (tmp_path / "x.ms").mkdir()
    result = runner.invoke(cli_image.image_app, ["ms-overview", str(tmp_path / "x.ms")])
    assert result.exit_code == 1
    assert "python-casacore" in result.output
