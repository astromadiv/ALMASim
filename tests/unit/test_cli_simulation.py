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


def _planned(monkeypatch, tmp_path, *args):
    """The runs `simulation run <args>` would make: (row, cube name, subdir, n_pix)."""
    planned = []

    def build(**kwargs):
        cube = kwargs["external_input_path"]
        planned.append(
            (kwargs["row_idx"], cube and cube.parent.name, kwargs["output_subdir_name"],
             kwargs["n_pix"])
        )  # fmt: skip
        return SimpleNamespace()

    monkeypatch.setattr(cli_simulation, "_build_simulation_params", build)
    monkeypatch.setattr(cli_simulation, "_run_simulation_stages", lambda *a, **k: None)
    monkeypatch.setattr(cli_simulation, "create_backend", lambda *a, **k: _DummyBackend())
    result = runner.invoke(
        cli.app, ["simulation", "run", "--output-dir", str(tmp_path / "out"), *args]
    )
    return result, planned


def _imaging_run(tmp_path):
    """Two windows of one execution block, its observation in row 1 of the metadata."""
    import numpy as np
    from astropy.io import fits

    eb = "uid___A002_X11a1c8d_X12c8e"
    for spw in ("SPW-39_FIELD-9", "SPW-41_FIELD-9"):
        cube = tmp_path / "cubes" / f"{eb}.ms.split" / spw / "wsclean-cube-model.fits"
        cube.parent.mkdir(parents=True)
        fits.PrimaryHDU(np.zeros((1, 2, 4, 4), dtype=np.float32)).writeto(cube)
    pd.DataFrame(
        {"uid": ["uid://A001/X362b/Xf37"], "filename": [f"2023.1.00494.S_{eb}.asdm.sdm.tar"]}
    ).to_csv(tmp_path / "resolved_products.csv", index=False)
    metadata = _write_metadata_csv(
        tmp_path,
        [{"member_ous_uid": "uid://A001/X1/X2"}, {"member_ous_uid": "uid://A001/X362b/Xf37"}],
    )
    return tmp_path / "cubes", metadata


def test_each_input_cube_runs_with_the_row_of_its_observation(monkeypatch, tmp_path):
    cubes, metadata = _imaging_run(tmp_path)
    result, planned = _planned(
        monkeypatch, tmp_path, "--metadata-path", metadata, "--input", str(cubes),
        "--save-mode", "fits",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    eb = "uid___A002_X11a1c8d_X12c8e"
    assert planned == [
        (1, "SPW-39_FIELD-9", f"sim_demo_{eb}_SPW-39_FIELD-9_0", None),
        (1, "SPW-41_FIELD-9", f"sim_demo_{eb}_SPW-41_FIELD-9_0", None),
    ]


def test_an_input_cube_sets_its_own_size(monkeypatch, tmp_path):
    cubes, metadata = _imaging_run(tmp_path)
    result, planned = _planned(
        monkeypatch, tmp_path, "--metadata-path", metadata, "--input", str(cubes),
        "--n-pix", "64",
    )  # fmt: skip
    assert result.exit_code == 2 and not planned
    assert "sets its own size" in result.output + (result.stderr or "")


def test_a_cube_that_names_no_observation_serves_the_chosen_row(monkeypatch, tmp_path):
    import numpy as np
    from astropy.io import fits

    cube = tmp_path / "anything.fits"
    fits.PrimaryHDU(np.zeros((2, 4, 4), dtype=np.float32)).writeto(cube)
    metadata = _write_metadata_csv(tmp_path, [{"member_ous_uid": "uid://A"}] * 3)
    result, planned = _planned(
        monkeypatch, tmp_path, "--metadata-path", metadata, "--input", str(cube),
        "--row-idx", "2",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert planned == [(2, tmp_path.name, None, None)]


def test_without_an_input_cube_the_size_defaults_as_before(monkeypatch, tmp_path):
    metadata = _write_metadata_csv(tmp_path, [{"member_ous_uid": "uid://A"}])
    result, planned = _planned(monkeypatch, tmp_path, "--metadata-path", metadata)
    assert result.exit_code == 0, result.output
    assert planned == [(0, None, None, 128)]


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


def test_an_ms_is_written_only_with_ms_export(monkeypatch, tmp_path):
    prepared = []
    monkeypatch.setattr(
        cli_simulation, "_prepare_ms_export", lambda out, casa: prepared.append((out, casa))
    )
    kwargs, result = _params_from_cli(monkeypatch, tmp_path)
    assert result.exit_code == 0, result.output
    assert not kwargs["ms_export"] and kwargs["ms_save_mode"] == "none" and not prepared
    kwargs, result = _params_from_cli(
        monkeypatch, tmp_path, "--ms-export", "--casa-data-root", str(tmp_path / "casa")
    )
    assert result.exit_code == 0, result.output
    assert kwargs["ms_export"] and kwargs["ms_save_mode"] == "msv2"
    assert kwargs["ms_export_dir"] is None  # <project>_<run>.ms, where the products go
    assert prepared == [((tmp_path / "out").resolve(), tmp_path / "casa")]


def test_runs_that_share_the_output_folder_write_their_own_ms(monkeypatch, tmp_path):
    """Two cubes of one row share its run index: each MS takes its run's name."""
    cubes, metadata = _imaging_run(tmp_path)
    monkeypatch.setattr(cli_simulation, "_prepare_ms_export", lambda out, casa: None)
    paths = []

    def build(**kwargs):
        paths.append(kwargs["ms_path"])
        return SimpleNamespace()

    monkeypatch.setattr(cli_simulation, "_build_simulation_params", build)
    monkeypatch.setattr(cli_simulation, "_run_simulation_stages", lambda *a, **k: None)
    monkeypatch.setattr(cli_simulation, "create_backend", lambda *a, **k: _DummyBackend())
    out = tmp_path / "out"
    args = ["simulation", "run", "--output-dir", str(out), "--metadata-path", metadata]
    result = runner.invoke(cli.app, [*args, "--input", str(cubes), "--ms-export"])
    assert result.exit_code == 0, result.output
    eb = "uid___A002_X11a1c8d_X12c8e"
    assert paths == [
        out.resolve() / f"sim_demo_{eb}_SPW-39_FIELD-9_0.ms",
        out.resolve() / f"sim_demo_{eb}_SPW-41_FIELD-9_0.ms",
    ]
    paths.clear()  # kept products: each run's MS goes in its own folder
    result = runner.invoke(
        cli.app, [*args, "--input", str(cubes), "--ms-export", "--save-mode", "fits"]
    )
    assert result.exit_code == 0, result.output
    assert paths == [None, None]


def _no_ms_writer(monkeypatch, *, tty: bool, answer: bool | None = None):
    import importlib.util

    find_spec = importlib.util.find_spec
    installed: set[str] = set()
    monkeypatch.setattr(
        cli_simulation.importlib.util,
        "find_spec",
        lambda name, *a: (
            (object() if name in installed else None)
            if name in ("casatools", "casacore")
            else find_spec(name, *a)
        ),
    )
    monkeypatch.setattr(cli_simulation.sys.stdin, "isatty", lambda: tty, raising=False)
    asked = []

    def confirm(question, default=False):
        asked.append(question)
        return answer

    monkeypatch.setattr(cli_simulation.typer, "confirm", confirm)
    return installed, asked


def test_without_an_ms_writer_a_job_says_how_to_get_one(monkeypatch, tmp_path, capsys):
    import pytest
    import typer

    _, asked = _no_ms_writer(monkeypatch, tty=False)
    with pytest.raises(typer.Exit) as exit_info:
        cli_simulation._prepare_ms_export(tmp_path, None)
    assert exit_info.value.exit_code == 1 and not asked
    printed = capsys.readouterr().err
    assert "uv sync --extra ms-casacore" in printed and "pip install python-casacore" in printed


def test_without_an_ms_writer_a_terminal_offers_to_install_one(monkeypatch, tmp_path):
    import pytest
    import typer

    installed, asked = _no_ms_writer(monkeypatch, tty=True, answer=False)
    with pytest.raises(typer.Exit):
        cli_simulation._prepare_ms_export(tmp_path, None)
    assert asked == ["Install python-casacore into this environment now?"]

    installed, asked = _no_ms_writer(monkeypatch, tty=True, answer=True)
    commands = []

    def run(command, check=False):
        commands.append(command)
        installed.add("casacore")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_simulation.subprocess, "run", run)
    monkeypatch.setattr(cli_simulation.shutil, "which", lambda name: "/bin/uv")
    cli_simulation._prepare_ms_export(tmp_path, None)  # installed: no exit
    assert commands == [
        ["/bin/uv", "pip", "install", "--python", cli_simulation.sys.executable, "python-casacore"]
    ]


def test_casatools_gets_its_runtime_data_before_any_run(monkeypatch, tmp_path):
    """Without it casatools fails to load, and python-casacore, loaded after it
    in the same process, aborts it (free(): invalid pointer)."""
    import importlib.util

    import almasim.services.casa.environment as environment
    import almasim.services.casa.runtime_data as runtime_data

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        cli_simulation.importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "casatools" else find_spec(name, *a),
    )
    configured, ensured = [], []
    monkeypatch.setattr(
        environment,
        "configure_casa_environment",
        lambda root, data: configured.append((root, data)),
    )
    monkeypatch.setattr(
        runtime_data, "ensure_casa_runtime_data", lambda data, logger_fn=None: ensured.append(data)
    )
    cli_simulation._prepare_ms_export(tmp_path, None)
    assert configured == [(tmp_path / ".casa-work", tmp_path / ".casa-data")]
    assert ensured == [tmp_path / ".casa-data"]
    configured.clear()
    cli_simulation._prepare_ms_export(tmp_path, tmp_path / "shared")
    assert configured == [(tmp_path / ".casa-work", (tmp_path / "shared").absolute())]
