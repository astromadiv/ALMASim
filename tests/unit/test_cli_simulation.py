"""Unit tests for ALMASim simulation CLI commands."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from almasim import cli
from almasim.cli import simulation as cli_simulation

runner = CliRunner()


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class _DummyBackend:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None


def _write_metadata_csv(tmp_path, rows: list[dict]) -> str:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return str(metadata_path)


def test_simulation_requires_metadata_path():
    """Simulation runs must require --metadata-path."""
    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "run",
        ],
    )

    assert result.exit_code == 2
    cleaned = _ANSI_ESCAPE_RE.sub("", result.output)
    assert "Missing option" in cleaned
    assert "--metadata-path" in cleaned


def test_simulation_invalid_backend(tmp_path):
    """Invalid backend values should fail fast."""
    metadata_path = _write_metadata_csv(tmp_path, [{"member_ous_uid": "uid://A"}])

    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "run",
            "--metadata-path",
            metadata_path,
            "--backend",
            "dask",
        ],
    )

    assert result.exit_code == 2
    assert "'dask' is not one of 'sync', 'local'" in " ".join(
        re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("│", " ").split()
    )


def test_simulation_run_csv_happy_path(monkeypatch, tmp_path):
    """Simulation run should execute staged workflow and print completion output."""
    metadata_path = _write_metadata_csv(tmp_path, [{"member_ous_uid": "uid://A"}])

    monkeypatch.setattr(cli_simulation.astro, "get_line_info", lambda main_dir: (100.0, None))
    monkeypatch.setattr(
        cli_simulation,
        "with_redshifts",
        lambda metadata, rest_frequency: metadata.assign(redshift=0.1, Freq=100.0),
    )

    monkeypatch.setattr(
        cli_simulation.SimulationParams,
        "from_metadata_row",
        lambda *args, **kwargs: SimpleNamespace(ml_dataset_path="/tmp/mock_ml.h5"),
    )
    monkeypatch.setattr(cli_simulation, "create_backend", lambda *args, **kwargs: _DummyBackend())
    monkeypatch.setattr(
        cli_simulation,
        "generate_clean_cube",
        lambda *args, **kwargs: SimpleNamespace(model_cube=SimpleNamespace(shape=(1, 2, 3))),
    )
    monkeypatch.setattr(
        cli_simulation,
        "simulate_observation",
        lambda *args, **kwargs: {
            "dirty_cube": SimpleNamespace(shape=(1, 2, 3)),
            "uv_mask_cube": SimpleNamespace(shape=(1, 2, 3)),
        },
    )
    monkeypatch.setattr(
        cli_simulation,
        "image_products",
        lambda results: {**results, "int_image_cube": "imaged"},
    )
    exported = []

    def fake_export(params, clean_stage, simulation_results, **kwargs):
        exported.append(simulation_results)
        return {"ml_dataset_path": "/tmp/mock_ml.h5", "dirty_cube": "dummy"}

    monkeypatch.setattr(cli_simulation, "export_results", fake_export)

    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "run",
            "--metadata-path",
            metadata_path,
        ],
    )

    assert result.exit_code == 0
    assert exported[0]["int_image_cube"] == "imaged", "image cubes are built before export"
    assert "All done. Simulated 1 run(s) across 1 row(s)." in result.output
    assert "ML shard written to: /tmp/mock_ml.h5" in result.output


def test_resolve_input_path_for_row_single_file(tmp_path):
    """A single input file should be reused for every metadata row."""
    input_file = tmp_path / "shared_model.fits"
    input_file.write_text("dummy", encoding="utf-8")
    row = pd.Series({"member_ous_uid": "uid://A", "group_ous_uid": "uid://G"})

    resolved = cli_simulation._resolve_input_path_for_row(input_file, row)

    assert resolved == input_file.resolve()


def test_resolve_input_path_for_row_directory_match(tmp_path):
    """Input directory should match files using member_ous_uid/group_ous_uid."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    matched = input_dir / "skymodel_uidA001X1X1_uidA001X1.fits"
    matched.write_text("dummy", encoding="utf-8")
    (input_dir / "other_model.fits").write_text("dummy", encoding="utf-8")
    row = pd.Series(
        {
            "member_ous_uid": "uid://A001/X1/X1",
            "group_ous_uid": "uid://A001/X1",
        }
    )

    resolved = cli_simulation._resolve_input_path_for_row(input_dir, row)

    assert resolved == matched.resolve()


def test_simulation_invalid_source_type_exits(tmp_path):
    """Invalid source type should fail with exit code 2."""
    metadata_path = _write_metadata_csv(tmp_path, [{"member_ous_uid": "uid://A"}])

    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "run",
            "--metadata-path",
            metadata_path,
            "--source-type",
            "not-a-source",
        ],
    )

    assert result.exit_code == 2
    assert "'not-a-source' is not one of" in " ".join(
        re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("│", " ").split()
    )


def test_the_cli_uses_the_simulation_service_directly():
    """No lazy forwarders: the CLI's names are the service's own objects."""
    import almasim.services.compute as compute
    import almasim.services.simulation as simulation

    assert cli_simulation.SimulationParams is simulation.SimulationParams
    assert cli_simulation.create_backend is compute.create_backend
    for name in ("generate_clean_cube", "simulate_observation", "image_products", "export_results"):
        assert getattr(cli_simulation, name) is getattr(simulation, name)


def test_a_row_index_simulates_exactly_that_row(monkeypatch, tmp_path):
    """--row-idx picks that CSV row (it once sampled one at random)."""
    rows = [
        {"member_ous_uid": f"uid://A{i}", "ALMA_source_name": f"S{i}", "Freq": f}
        for i, f in enumerate([230.0, 100.0, 345.0])
    ]
    metadata_path = _write_metadata_csv(tmp_path, rows)
    monkeypatch.setattr(cli_simulation.astro, "get_line_info", lambda main_dir: (400.0, None))
    picked = []

    def from_row(row, **kwargs):
        picked.append((row["member_ous_uid"], round(float(row["redshift"]), 3)))
        raise SystemExit(0)

    monkeypatch.setattr(cli_simulation.SimulationParams, "from_metadata_row", from_row)
    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "run",
            "--metadata-path",
            metadata_path,
            "--row-idx",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert picked == [("uid://A1", 3.0)]  # 400 GHz line seen at 100 GHz


def test_a_row_with_no_line_above_it_is_refused(monkeypatch, tmp_path):
    metadata_path = _write_metadata_csv(
        tmp_path, [{"member_ous_uid": "uid://A", "ALMA_source_name": "S", "Freq": 500.0}]
    )
    monkeypatch.setattr(cli_simulation.astro, "get_line_info", lambda main_dir: (400.0, None))
    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "run",
            "--metadata-path",
            metadata_path,
            "--row-idx",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    printed = result.output + (result.stderr or "")
    text = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", printed).replace("│", " ").split())
    assert "no catalogue line at or above its frequency" in text


def _params_from_cli(monkeypatch, tmp_path, *extra: str):
    """The keyword arguments the CLI builds a row's SimulationParams from, and the result."""
    metadata_path = _write_metadata_csv(
        tmp_path,
        [{"member_ous_uid": "uid://A001/X1/X1", "ALMA_source_name": "S", "Freq": 100.0}],
    )
    monkeypatch.setattr(cli_simulation.astro, "get_line_info", lambda main_dir: (400.0, None))
    built = []

    def from_row(row, **kwargs):
        built.append(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(cli_simulation.SimulationParams, "from_metadata_row", from_row)
    result = runner.invoke(
        cli.app,
        [
            "simulation",
            "--metadata-path",
            metadata_path,
            "--row-idx",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
            *extra,
        ],
    )
    return (built[0] if built else None), result


def test_a_background_needs_its_mode(monkeypatch, tmp_path):
    """--background-level once did nothing: the CLI never set background_mode."""
    kwargs, result = _params_from_cli(
        monkeypatch, tmp_path, "--background-mode", "combined", "--background-level", "2.5"
    )
    assert result.exit_code == 0, result.output
    assert (kwargs["background_mode"], kwargs["background_level"]) == ("combined", 2.5)

    kwargs, _ = _params_from_cli(monkeypatch, tmp_path, "--background-mode", "dusty_diffuse")
    assert (kwargs["background_mode"], kwargs["background_level"]) == ("dusty_diffuse", 1.0)

    kwargs, _ = _params_from_cli(monkeypatch, tmp_path)
    assert kwargs["background_mode"] == "none"

    kwargs, result = _params_from_cli(monkeypatch, tmp_path, "--background-level", "2")
    assert kwargs is None and result.exit_code == 2
    assert "--background-mode" in result.output + (result.stderr or "")

    # 0, the old default, still runs without a mode.
    kwargs, result = _params_from_cli(monkeypatch, tmp_path, "--background-level", "0")
    assert result.exit_code == 0, result.output
    assert kwargs["background_mode"] == "none"


def test_external_components_read_their_input_as_a_table(monkeypatch, tmp_path):
    """--input was passed as a FITS sky model, which external-components never reads."""
    table = tmp_path / "components.ecsv"
    table.write_text("dummy", encoding="utf-8")
    kwargs, result = _params_from_cli(
        monkeypatch, tmp_path, "--source-type", "external-components", "--input", str(table)
    )
    assert result.exit_code == 0, result.output
    assert kwargs["source_type"] == "external-components"
    assert kwargs["external_component_table_path"] == table.resolve()
    assert kwargs["external_skymodel_path"] is None

    # A folder holds FITS cubes to match to rows, not tables.
    kwargs, result = _params_from_cli(
        monkeypatch, tmp_path, "--source-type", "external-components", "--input", str(tmp_path)
    )
    assert kwargs is None and result.exit_code == 2
    assert "one component table" in result.output + (result.stderr or "")


def test_a_fits_input_stays_a_sky_model(monkeypatch, tmp_path):
    cube = tmp_path / "cube.fits"
    cube.write_text("dummy", encoding="utf-8")
    kwargs, result = _params_from_cli(monkeypatch, tmp_path, "--input", str(cube))
    assert result.exit_code == 0, result.output
    assert kwargs["source_type"] == "external-fits-cube"
    assert kwargs["external_skymodel_path"] == cube.resolve()
    assert kwargs["external_component_table_path"] is None
