"""CLI usage errors exit 2 and failures exit 1, with a message on stderr (MADIV-28 T3.3)."""

from __future__ import annotations

import re

import pandas as pd
import pytest
from typer.testing import CliRunner

from almasim import cli
from almasim.cli import clean as cli_clean
from almasim.cli import metadata as cli_metadata
from almasim.cli import products as cli_products

pytestmark = pytest.mark.unit
runner = CliRunner()


def _invoke(*args):
    return runner.invoke(cli.app, [str(a) for a in args])


def _plain(output: str) -> str:
    """Typer's error panel as one line, without colours (CI forces them) or box drawing."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", output).replace("│", " ").split())


@pytest.mark.parametrize("command", ["extract", "unpack", "calibrate"])
def test_archive_stages_refuse_an_unknown_backend(command):
    result = _invoke("visibilities", command, "--postprocess-backend", "dask")
    # The choices are an Enum (MADIV-28 T3.2): typer rejects the value itself.
    assert result.exit_code == 2
    assert "'dask' is not one of 'sync', 'slurm'" in _plain(result.output)


def test_extract_needs_an_existing_source_with_archives(tmp_path):
    result = _invoke("visibilities", "extract", "--source-root", tmp_path / "missing")
    assert result.exit_code == 2 and "--source-root is not a directory" in result.output
    result = _invoke("visibilities", "extract", "--source-root", tmp_path)
    assert result.exit_code == 1 and "No .tar/.tgz archives found" in result.output


def test_resolve_needs_member_uids(tmp_path):
    result = _invoke("products", "resolve")
    assert result.exit_code == 2
    csv = tmp_path / "meta.csv"
    pd.DataFrame({"source": ["a"]}).to_csv(csv, index=False)
    result = _invoke("products", "resolve", "--metadata-csv", csv)
    assert result.exit_code == 2 and "does not contain member_ous_uid" in result.output


def test_download_validates_its_options_and_reports_empty_results(monkeypatch):
    result = _invoke("products", "download", "--member-ous-uid", "uid://A", "--product-filter", "x")
    assert result.exit_code == 2 and "'x' is not one of 'all'" in _plain(result.output)
    result = _invoke(
        "products", "download", "--member-ous-uid", "uid://A", "--postprocess-backend", "dask"
    )
    assert result.exit_code == 2 and "is not one of 'sync', 'slurm'" in _plain(result.output)
    monkeypatch.setattr(cli_products, "resolve_products", lambda uids, **kw: [])
    result = _invoke("products", "download", "--member-ous-uid", "uid://A", "--yes")
    assert result.exit_code == 1


def test_metadata_query_refuses_unknown_array_types_and_reports_no_rows(monkeypatch):
    result = _invoke("metadata", "query", "--array-type", "9m")
    assert result.exit_code == 2 and "--array-type" in result.output
    monkeypatch.setattr(cli_metadata, "query_metadata_by_science", lambda **kw: pd.DataFrame())
    result = _invoke("metadata", "query", "--science-keyword", "Galaxies", "--yes")
    assert result.exit_code == 1 and "No metadata rows matched" in result.output


def test_simulation_run_checks_its_inputs(tmp_path):
    csv = tmp_path / "meta.csv"
    pd.DataFrame({"member_ous_uid": ["uid://A"]}).to_csv(csv, index=False)
    cases = [
        (["--save-mode", "pickle"], "'pickle' is not one of 'memory', 'npz', 'h5', 'fits'"),
        (["--input", tmp_path / "none.fits"], "--input not found"),
        (["--kaggle-token", tmp_path / "kaggle.json"], "--kaggle-token not found"),
        (["--noise-model", "stormy"], "not a released model"),
        (["--corruptions", '{"gain_amplitude": 0.1}'], "unknown corruption parameter"),
        (["--noise-model", "alma", "--noise-without", "wind"], "unknown noise component"),
        (["--noise-only", "radiometer"], "select components of a --noise-model"),
    ]
    for extra, message in cases:
        result = _invoke("simulation", "run", "--metadata-path", csv, *extra)
        assert result.exit_code == 2 and message in _plain(result.output), (extra, result.output)
    result = _invoke("simulation", "run", "--metadata-path", tmp_path / "missing.csv")
    assert result.exit_code == 2 and "--metadata-path not found" in result.output


def test_clean_needs_arguments_and_reports_a_wsclean_that_cannot_start(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import typer

    with pytest.raises(typer.Exit) as info:
        cli_clean.clean_command(
            SimpleNamespace(args=[]), wsclean_bin="wsclean", cwd=None, dry_run=False
        )
    assert info.value.exit_code == 2

    def broken(*args, **kwargs):
        raise OSError("exec format error")

    monkeypatch.setattr(cli_clean.subprocess, "run", broken)
    wsclean = tmp_path / "wsclean"
    wsclean.write_text("#!/bin/sh\n")
    wsclean.chmod(0o755)
    monkeypatch.setenv("ALMASIM_WSCLEAN_BIN", str(wsclean))
    result = _invoke("clean", "--", "-name", "x", "in.ms")
    assert result.exit_code == 1 and "Failed to start WSClean" in result.output


def test_flag_summary_refuses_a_path_that_is_no_measurement_set(tmp_path):
    not_ms = tmp_path / "notes.txt"
    not_ms.write_text("x")
    result = _invoke("visibilities", "flag-summary", not_ms, "--output", tmp_path / "s.csv")
    assert result.exit_code == 1 and "Not a MeasurementSet or folder" in result.output


def test_old_option_spellings_still_work_with_a_warning(tmp_path):
    """--postprocess-backend and friends are aliases of the new names (MADIV-28 T3.9)."""
    result = _invoke(
        "visibilities", "extract", "--source-root", tmp_path, "--postprocess-backend", "sync"
    )
    assert result.exit_code == 1 and "No .tar/.tgz archives found" in result.output
    assert "--postprocess-backend is deprecated and will be removed; use --backend" in _plain(
        result.output
    )
    result = _invoke("visibilities", "extract", "--source-root", tmp_path, "--backend", "sync")
    assert result.exit_code == 1 and "deprecated" not in result.output


def test_deprecated_flag_warnings_cover_every_old_spelling():
    from almasim.cli._options import DEPRECATED_FLAGS, deprecated_flag_warnings

    warnings = deprecated_flag_warnings(["--slurm-n-jobs", "2", "--output-root=/x", "--backend"])
    assert warnings == [
        "WARNING: --slurm-n-jobs is deprecated and will be removed; use --slurm-workers.",
        "WARNING: --output-root is deprecated and will be removed; use --output-dir.",
    ]
    assert set(DEPRECATED_FLAGS.values()) == {
        "--backend",
        "--scheduler-host",
        "--slurm-workers",
        "--output-dir",
        "--noise-model",
    }
