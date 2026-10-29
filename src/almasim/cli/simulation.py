"""Simulation commands for the ALMASim CLI."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Any

import pandas as pd
import typer
from tqdm.auto import tqdm

from ..services import astro
from ..services.astro.spectral import with_redshifts
from ..services.compute import create_backend
from ..services.external_skymodel import EXTERNAL_SOURCE_TYPES, is_external_source_type
from ..services.noise_model import NoiseModel, resolve_noise_model
from ..services.simulation import (
    SimulationParams,
    export_results,
    generate_clean_cube,
    image_products,
    simulate_observation,
)
from ..services.simulation.background import BACKGROUND_MODES
from ..services.simulation.input_cubes import (
    cube_label,
    find_input_cubes,
    match_input_cubes,
    observations_by_execution_block,
    products_csvs_near,
)
from ..skymodels.registry import SKYMODELS
from ._shared import fail, usage_error


class InstrumentGeometry(StrEnum):
    """How a simulation sets its beam, field of view and cell (MADIV-28 T2.11)."""

    legacy = "legacy"
    planner = "planner"


simulation_app = typer.Typer(
    help="Simulation workflow commands.",
    no_args_is_help=True,
)

_SOURCE_TYPES = [name for name, spec in SKYMODELS.items() if spec.in_cli and not spec.external]
_SAVE_MODES = ["memory", "npz", "h5", "fits"]
_BACKEND_TYPES = ["sync", "local"]
# "ducc0" is disabled until its uv geometry is fixed (MADIV-28 T1.3, T3.7).
_IMAGING_ALGORITHMS = ["legacy"]

# The choices as enums, so typer checks them and the console's catalogue can
# list them (MADIV-28 S2.2). Members are str: the code receiving them is unchanged.
SourceType = StrEnum(  # type: ignore[misc]
    "SourceType", {name: name for name in _SOURCE_TYPES + sorted(EXTERNAL_SOURCE_TYPES)}
)
SaveMode = StrEnum("SaveMode", {name: name for name in _SAVE_MODES})  # type: ignore[misc]
SimulationBackend = StrEnum(  # type: ignore[misc]
    "SimulationBackend", {name: name for name in _BACKEND_TYPES}
)
ImagingAlgorithm = StrEnum(  # type: ignore[misc]
    "ImagingAlgorithm", {name: name for name in _IMAGING_ALGORITHMS}
)
BackgroundMode = StrEnum(  # type: ignore[misc]
    "BackgroundMode", {name: name for name in BACKGROUND_MODES}
)
# The cube size without an --input cube (which sets its own).
DEFAULT_N_PIX = 128
DEFAULT_N_CHANNELS = 32


def _path_with_suffix(path: Path, *parts: Any) -> Path:
    suffix = "_".join(str(part) for part in parts)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _update_progress(progress_bar: tqdm, target: int) -> None:
    clamped = max(0, min(100, int(target)))
    if clamped > progress_bar.n:
        progress_bar.update(clamped - progress_bar.n)


def _progress_callback(progress_bar: tqdm, start: int, span: int):
    def callback(value: int | float) -> None:
        _update_progress(progress_bar, start + int(round((float(value) / 100.0) * span)))

    return callback


def _progress_logger(progress_bar: tqdm):
    def logger(message: str) -> None:
        progress_bar.write(str(message))

    return logger


def _run_stage_with_feedback(
    progress_bar: tqdm,
    *,
    label: str,
    work,
    smooth_target: int | None = None,
):
    stop_event = Event()
    spinner_frames = "|/-\\"

    def refresh() -> None:
        spinner_index = 0
        while not stop_event.wait(0.2):
            progress_bar.set_postfix_str(
                f"{label} {spinner_frames[spinner_index % len(spinner_frames)]}"
            )
            spinner_index += 1
            if smooth_target is not None and progress_bar.n < smooth_target:
                progress_bar.update(1)
                sleep(0.8)

    thread = Thread(target=refresh, daemon=True)
    thread.start()
    try:
        return work()
    finally:
        stop_event.set()
        thread.join(timeout=1)
        progress_bar.set_postfix_str("")


def _build_simulation_params(
    *,
    metadata: pd.DataFrame,
    row_idx: int,
    run_idx: int,
    main_dir: Path,
    output_dir: Path,
    project_name: str,
    source_type: str,
    save_mode: str,
    persist_standard_outputs: bool,
    ml_shard_path: Path,
    n_pix: int | None,
    n_channels: int | None,
    n_lines: int | None,
    imaging_algorithm: str,
    background_mode: str = "none",
    background_level: float,
    background_seed: int | None,
    seed: int | None,
    instrument_geometry: str = "legacy",
    inject_serendipitous: bool,
    output_subdir_name: str | None,
    external_input_path: Path | None,
    kaggle_token: Path | None = None,
    noise_model: NoiseModel | str | None = None,
    executions: int | None = None,
    ms_export: bool = False,
    ms_path: Path | None = None,
) -> SimulationParams:
    if not 0 <= row_idx < len(metadata):
        usage_error(f"--row-idx {row_idx} is out of range for {len(metadata)} metadata rows")
    # Exactly that row of the CSV, with the redshift of its catalogue line.
    rest_frequency, _ = astro.get_line_info(main_dir)
    row = with_redshifts(metadata.iloc[[row_idx]], rest_frequency).iloc[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    component_table = source_type == "external-components"
    effective_source_type = (
        source_type
        if external_input_path is None or is_external_source_type(source_type)
        else "external-fits-cube"
    )
    if not is_external_source_type(effective_source_type) and not row["redshift"] >= 0:
        usage_error(
            f"row {row_idx} ({row.get('ALMA_source_name', '?')}, {row['Freq']} GHz): no "
            "catalogue line at or above its frequency, so no redshift places one there"
        )

    return SimulationParams.from_metadata_row(
        row,
        idx=run_idx,
        main_dir=main_dir,
        output_dir=output_dir,
        tng_dir=output_dir / "tng",
        galaxy_zoo_dir=output_dir / "galaxy_zoo",
        hubble_dir=output_dir / "hubble",
        project_name=project_name,
        source_type=effective_source_type,
        save_mode=save_mode,
        persist=persist_standard_outputs,
        ml_dataset_path=ml_shard_path,
        output_subdir_name=output_subdir_name,
        # A component table is read as a table, not as a FITS image.
        external_skymodel_path=None if component_table else external_input_path,
        external_component_table_path=external_input_path if component_table else None,
        n_pix=n_pix,
        n_channels=n_channels,
        n_lines=n_lines,
        imaging_algorithm=imaging_algorithm,
        background_mode=background_mode,
        background_level=background_level,
        background_seed=background_seed,
        seed=seed,
        instrument_geometry=instrument_geometry,
        inject_serendipitous=inject_serendipitous,
        kaggle_token=kaggle_token,
        noise_model=noise_model,
        executions=executions,
        # --ms-export; ms_path: where, when the run's own name is not enough.
        ms_export=ms_export,
        ms_save_mode="msv2" if ms_export else "none",
        ms_export_dir=str(ms_path) if ms_export and ms_path is not None else None,
    )


MS_WRITER_HELP = (
    "--ms-export needs casatools or python-casacore in the environment almasim runs in:\n"
    "  uv sync --extra ms-casacore    (in an ALMASim checkout; --extra casa for casatools)\n"
    "  pip install python-casacore    (an installed ALMASim)"
)


def _ms_writer_installed() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in ("casatools", "casacore"))


def _install_python_casacore() -> bool:
    """python-casacore installed into this environment (uv's pip when there is
    uv, since uv's environments have no pip); whether it can now be imported."""
    uv = shutil.which("uv")
    command = (
        [uv, "pip", "install", "--python", sys.executable, "python-casacore"]
        if uv
        else [sys.executable, "-m", "pip", "install", "python-casacore"]
    )
    typer.echo(f"Running: {' '.join(command)}")
    if subprocess.run(command, check=False).returncode != 0:
        return False
    importlib.invalidate_caches()
    return _ms_writer_installed()


def _prepare_ms_export(output_dir: Path, casa_data_root: Path | None) -> None:
    """Before any run: an MS writer, installed on request when there is none,
    and casatools' runtime data. Without its data casatools fails to load, and
    python-casacore, loaded after it in the same process, aborts it."""
    if not _ms_writer_installed():
        typer.echo(MS_WRITER_HELP, err=True)
        if not sys.stdin.isatty():
            fail("No MS writer: install one as above, or run without --ms-export.")
        if not typer.confirm("Install python-casacore into this environment now?", default=False):
            fail("No MS writer: install one as above, or run without --ms-export.")
        if not _install_python_casacore():
            fail("Installing python-casacore failed (see above); install it as above.")
    if importlib.util.find_spec("casatools") is None:
        return  # python-casacore writes the MS and needs no runtime data
    from ..services.casa.environment import configure_casa_environment
    from ..services.casa.runtime_data import ensure_casa_runtime_data

    casa_data = (
        casa_data_root.expanduser().absolute()
        if casa_data_root is not None
        else output_dir / ".casa-data"
    )
    configure_casa_environment(output_dir / ".casa-work", casa_data)
    ensure_casa_runtime_data(casa_data, logger_fn=typer.echo)


@dataclass(frozen=True)
class SimulationRunOptions:
    """Every option of ``simulation run`` (MADIV-28 T3.8)."""

    metadata_path: Path
    row_idx: int | None
    project_name: str
    output_dir: Path
    input_path: Path | None
    products_csv: Path | None
    main_dir: Path
    source_type: str
    n_pix: int | None
    n_channels: int | None
    n_lines: int | None
    seed: int | None
    simulations_per_row: int
    imaging_algorithm: str
    robust: float
    instrument_geometry: InstrumentGeometry
    background_mode: str
    background_level: float | None
    background_seed: int | None
    inject_serendipitous: bool
    save_mode: str
    persist_standard_outputs: bool
    ml_shard_path: Path
    ms_export: bool
    casa_data_root: Path | None
    backend: str
    n_workers: int
    kaggle_token: Path | None
    noise_model: NoiseModel | str | None
    noise_only: str | None
    noise_without: str | None
    executions: int | None


@simulation_app.command("run")
def simulation_run(
    metadata_path: Path = typer.Option(
        ...,
        "--metadata-path",
        help="CSV file with pre-fetched ALMA metadata rows (from 'almasim metadata').",
        rich_help_panel="Observation",
    ),
    row_idx: int | None = typer.Option(
        None,
        "--row-idx",
        min=0,
        help="Single metadata row index to simulate. Omit to simulate all rows.",
        rich_help_panel="Observation",
    ),
    project_name: str = typer.Option(
        "sim_demo",
        "--project-name",
        help="Project name used in outputs.",
        rich_help_panel="Output",
    ),
    output_dir: Path = typer.Option(
        Path.home() / "almasim_outputs" / "simulation",
        "--output-dir",
        help="Directory for simulation outputs.",
        rich_help_panel="Output",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help=(
            "A sky-model cube, or a folder of them (an imaging run's: each spectral"
            " window's wsclean-cube-model.fits). Each cube is simulated with the"
            " metadata row of the observation its path names (an execution block,"
            " looked up in --products-csv, or a member OUS), on its own grid: size,"
            " pixel and frequency axis from its header. A file whose path names no"
            " observation is used for the chosen rows."
        ),
        rich_help_panel="Sky model",
    ),
    products_csv: Path | None = typer.Option(
        None,
        "--products-csv",
        help=(
            "Resolved products (almasim products resolve) that map execution blocks"
            " to observations, for --input (default: the *products*.csv files next to"
            " --metadata-path)."
        ),
        rich_help_panel="Sky model",
    ),
    main_dir: Path = typer.Option(
        Path(__file__).resolve().parents[1],
        "--main-dir",
        help="Path to ALMASim data directory (defaults to src/almasim).",
        rich_help_panel="Advanced",
    ),
    source_type: SourceType = typer.Option(
        SourceType("point"),
        "--source-type",
        help="Sky model family (the external ones take their cube from --input).",
        case_sensitive=False,
        rich_help_panel="Sky model",
    ),
    n_pix: int | None = typer.Option(
        None,
        "--n-pix",
        min=1,
        help=f"Spatial cube size (default {DEFAULT_N_PIX}; an --input cube sets its own).",
        rich_help_panel="Sky model",
    ),
    n_channels: int | None = typer.Option(
        None,
        "--n-channels",
        min=1,
        help=f"Channel count (default {DEFAULT_N_CHANNELS}; an --input cube sets its own).",
        rich_help_panel="Sky model",
    ),
    n_lines: int | None = typer.Option(
        None,
        "--n-lines",
        min=1,
        help="Number of spectral lines to inject. Omit to use metadata default.",
        rich_help_panel="Sky model",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help=(
            "Base random seed for reproducible simulations. When running multiple "
            "simulations, each run uses seed + run_index."
        ),
        rich_help_panel="Advanced",
    ),
    simulations_per_row: int = typer.Option(
        1,
        "--simulations-per-row",
        min=1,
        help="How many simulations to run for each selected metadata row.",
        rich_help_panel="Observation",
    ),
    imaging_algorithm: ImagingAlgorithm = typer.Option(
        ImagingAlgorithm("legacy"),
        "--imaging-algorithm",
        help="Channel imaging algorithm.",
        case_sensitive=False,
        rich_help_panel="Imaging",
    ),
    robust: float = typer.Option(
        0.0,
        "--robust",
        help="Robust weighting parameter for imaging.",
        rich_help_panel="Imaging",
    ),
    instrument_geometry: InstrumentGeometry = typer.Option(
        InstrumentGeometry.legacy,
        "--instrument-geometry",
        case_sensitive=False,
        help=(
            "Beam, field of view and cell: legacy (the metadata FOV, beam / 5) or planner "
            "(the imaging planner's: the primary beam's FOV, beam / 8)."
        ),
        rich_help_panel="Observation",
    ),
    background_mode: BackgroundMode = typer.Option(
        BackgroundMode("none"),
        "--background-mode",
        help=(
            "An astrophysical background added to the sky: blank_field_dsfg (faint dusty "
            "galaxies), dusty_diffuse (a smooth dusty field), combined (both) or none."
        ),
        case_sensitive=False,
        rich_help_panel="Sky model",
    ),
    background_level: float | None = typer.Option(
        None,
        "--background-level",
        min=0.0,
        help=(
            "Strength of the --background-mode background, relative to the continuum "
            "sensitivity (default 1; 0 adds none)."
        ),
        rich_help_panel="Sky model",
    ),
    background_seed: int | None = typer.Option(
        None,
        "--background-seed",
        help="Random seed for background sky generation.",
        rich_help_panel="Advanced",
    ),
    inject_serendipitous: bool = typer.Option(
        False,
        "--inject-serendipitous",
        help="Add a few random serendipitous sources around the target.",
        rich_help_panel="Sky model",
    ),
    save_mode: SaveMode = typer.Option(
        SaveMode("memory"),
        "--save-mode",
        help="Output save mode.",
        case_sensitive=False,
        rich_help_panel="Output",
    ),
    persist_standard_outputs: bool = typer.Option(
        False,
        "--persist-standard-outputs",
        help="Persist standard outputs in addition to ML shard.",
        rich_help_panel="Output",
    ),
    ml_shard_path: Path = typer.Option(
        Path.home() / "almasim_outputs" / "ddrm_training_sample.h5",
        "--ml-shard-path",
        help="Destination HDF5 shard path.",
        rich_help_panel="Output",
    ),
    ms_export: bool = typer.Option(
        False,
        "--ms-export",
        help=(
            "Also write each run's MeasurementSet, <project-name>_<run>.ms in --output-dir "
            "(in the run's own folder when there are several runs and products are kept), "
            "with any --save-mode: Stokes I in both hands, with the noise model's noise "
            "and WEIGHT. Needs casatools or python-casacore."
        ),
        rich_help_panel="Output",
    ),
    casa_data_root: Path | None = typer.Option(
        None,
        "--casa-data-root",
        help=(
            "CASA runtime data for writing the MS with casatools (default "
            "<output-dir>/.casa-data, downloaded once when missing). python-casacore "
            "alone needs none."
        ),
        rich_help_panel="Output",
    ),
    backend: SimulationBackend = typer.Option(
        SimulationBackend("sync"),
        "--backend",
        help=(
            "How the channels are imaged: sync, one after another in one process;"
            " local, in parallel worker processes on this node (--n-workers)."
        ),
        case_sensitive=False,
        rich_help_panel="Cluster",
    ),
    n_workers: int = typer.Option(
        1,
        "--n-workers",
        min=1,
        help="Worker processes for --backend local; give the job as many CPUs.",
        rich_help_panel="Cluster",
    ),
    kaggle_token: Path | None = typer.Option(
        None,
        "--kaggle-token",
        help=(
            "Path to a Kaggle credentials file (JSON with username/key, e.g. "
            "kaggle.json or access_token) or the directory containing it, used to "
            "download morphology images for the galaxy-zoo / hubble-100 source "
            "types. Defaults to the standard Kaggle locations if omitted."
        ),
        rich_help_panel="Sky model",
    ),
    noise_model: str | None = typer.Option(
        None,
        "--noise-model",
        "--corruptions",
        help=(
            "What the visibilities carry besides white thermal noise: a released noise "
            "model ('alma', the newest version, or 'alma@1': radiometer noise following "
            "ALMA's Tsys, correlated channels, antenna noise ripple and, on bright "
            "sources, the gain, atmospheric phase and baseline errors calibration "
            "leaves, as measured on real calibrated data; 'none'), a JSON object or a "
            "JSON file of a noise model. 'almasim noise show' lists the models."
        ),
        rich_help_panel="Noise",
    ),
    noise_only: str | None = typer.Option(
        None,
        "--noise-only",
        help=(
            "Apply only these components of the noise model (comma-separated, e.g. "
            "'radiometer,channel_correlation'); 'almasim noise show' lists them."
        ),
        rich_help_panel="Noise",
    ),
    noise_without: str | None = typer.Option(
        None,
        "--noise-without",
        help="Apply the noise model without these components (comma-separated).",
        rich_help_panel="Noise",
    ),
    executions: int | None = typer.Option(
        None,
        "--executions",
        min=1,
        help=(
            "How many of the MOUS's execution blocks to simulate (default: all). A "
            "row's Int.Time is the MOUS's total time on source; it is observed in "
            "blocks (the row's n_executions, else blocks of at most 3000 s), each a "
            "track near transit on its own day. 1 gives a single execution block."
        ),
        rich_help_panel="Observation",
    ),
) -> None:
    """Run staged ALMASim simulation from a pre-queried metadata CSV."""
    options = _validated_simulation_options(SimulationRunOptions(**locals()))
    o = options
    metadata = pd.read_csv(o.metadata_path)
    total_rows = len(metadata)
    typer.echo(f"Metadata rows available: {total_rows}")

    runs = _planned_runs(o, metadata)
    total_runs = len(runs) * o.simulations_per_row
    rows = sorted({idx for idx, _ in runs})
    typer.echo(f"Rows to simulate: {len(rows)}")
    if o.input_path is not None:
        typer.echo(f"Input cubes: {len({cube for _, cube in runs})}")
    typer.echo(f"Simulations per row: {o.simulations_per_row}")
    typer.echo(f"Total simulations: {total_runs}")

    ml_shard_resolved = o.ml_shard_path.expanduser().resolve()
    persist_outputs = o.persist_standard_outputs or o.save_mode != "memory"
    backend_kwargs = {"n_workers": o.n_workers} if o.backend == "local" else {}
    output_dir = o.output_dir.expanduser().resolve()
    if o.ms_export:
        _prepare_ms_export(output_dir, o.casa_data_root)

    with create_backend(o.backend, **backend_kwargs) as compute_backend:
        run_number = 0
        for idx, cube in runs:
            # Each run's own names: by row, and by cube when there are input cubes.
            label = f"row{idx:04d}" if cube is None else f"row{idx:04d}_{cube_label(cube)}"
            for simulation_idx in range(o.simulations_per_row):
                run_number += 1
                typer.echo(
                    f"\n--- Simulation {run_number}/{total_runs}: row {idx + 1}/{total_rows} "
                    f"(index {idx}){f' with {cube}' if cube is not None else ''}, "
                    f"repeat {simulation_idx + 1}/{o.simulations_per_row} ---"
                )
                run_idx = idx * o.simulations_per_row + simulation_idx
                run_seed = o.seed + run_idx if o.seed is not None else None
                if run_seed is not None:
                    typer.echo(f"Seed: {run_seed}")
                run_name = (
                    f"{o.project_name}_{run_idx}"
                    if cube is None
                    else f"{o.project_name}_{cube_label(cube)}_{simulation_idx}"
                )
                params = _build_simulation_params(
                    metadata=metadata,
                    row_idx=idx,
                    run_idx=run_idx,
                    main_dir=o.main_dir.expanduser().resolve(),
                    output_dir=output_dir,
                    project_name=o.project_name,
                    source_type=o.source_type,
                    save_mode=o.save_mode,
                    persist_standard_outputs=persist_outputs,
                    ml_shard_path=(
                        _path_with_suffix(ml_shard_resolved, label, f"sim{simulation_idx:03d}")
                        if total_runs > 1
                        else ml_shard_resolved
                    ),
                    n_pix=o.n_pix,
                    n_channels=o.n_channels,
                    n_lines=o.n_lines,
                    imaging_algorithm=o.imaging_algorithm,
                    background_mode=o.background_mode,
                    background_level=1.0 if o.background_level is None else o.background_level,
                    background_seed=(
                        o.background_seed if o.background_seed is not None else run_seed
                    ),
                    seed=run_seed,
                    instrument_geometry=str(o.instrument_geometry),
                    inject_serendipitous=o.inject_serendipitous,
                    output_subdir_name=run_name if persist_outputs and total_runs > 1 else None,
                    external_input_path=cube,
                    kaggle_token=o.kaggle_token,
                    noise_model=o.noise_model,
                    executions=o.executions,
                    ms_export=o.ms_export,
                    # Several runs whose products are not kept share --output-dir:
                    # each MS by its run's name (two cubes of one row share an index).
                    ms_path=(
                        output_dir / f"{run_name}.ms"
                        if not persist_outputs and total_runs > 1
                        else None
                    ),
                )
                _run_simulation_stages(
                    o, params, compute_backend, label=f"Simulation {run_number}/{total_runs}"
                )

    typer.echo(f"\nAll done. Simulated {total_runs} run(s) across {len(rows)} row(s).")


def _planned_runs(o: SimulationRunOptions, metadata: pd.DataFrame) -> list[tuple[int, Path | None]]:
    """Every run's metadata row and input cube: the chosen rows, or each input
    cube with the rows of the observation it names (--row-idx keeps one row)."""
    rows = [o.row_idx] if o.row_idx is not None else list(range(len(metadata)))
    if o.input_path is None:
        return [(i, None) for i in rows]
    cubes = find_input_cubes(o.input_path)
    if not cubes:
        usage_error(f"--input: no FITS cube under {o.input_path}")
    products = [o.products_csv] if o.products_csv else products_csvs_near(o.metadata_path)
    matches, unmatched = match_input_cubes(
        cubes, metadata, observations_by_execution_block(products)
    )
    if not matches:
        if len(cubes) == 1:  # a cube whose path names no observation: for the chosen rows
            return [(i, cubes[0]) for i in rows]
        usage_error(
            f"--input: none of its {len(cubes)} cubes names an observation in the metadata"
            f" (by member OUS, or by execution block in {len(products)} products CSV(s))"
        )
    if unmatched:
        typer.echo(f"{len(unmatched)} cube(s) match no metadata row, e.g. {unmatched[0]}")
    runs = [(m.row_idx, m.cube) for m in matches if o.row_idx is None or m.row_idx == o.row_idx]
    if not runs:
        usage_error(f"--input: no cube matches metadata row {o.row_idx}")
    return runs


def _validated_input(o: SimulationRunOptions) -> SimulationRunOptions:
    """``o`` with the cube size the --input (or its absence) gives; a usage error
    for an input it cannot use."""
    if o.input_path is not None and not o.input_path.expanduser().exists():
        usage_error(f"--input not found: {o.input_path}")
    if o.products_csv is not None and not o.products_csv.expanduser().exists():
        usage_error(f"--products-csv not found: {o.products_csv}")
    if o.source_type == "external-components" and (
        o.input_path is None or not o.input_path.expanduser().is_file()
    ):
        usage_error("--source-type external-components takes one component table as --input")
    # A cube sets its own grid; a component table has none.
    input_cube = o.input_path is not None and o.source_type != "external-components"
    if input_cube and (o.n_pix is not None or o.n_channels is not None):
        usage_error("an --input cube sets its own size: drop --n-pix and --n-channels")
    if input_cube:
        return o
    return replace(
        o,
        n_pix=o.n_pix if o.n_pix is not None else DEFAULT_N_PIX,
        n_channels=o.n_channels if o.n_channels is not None else DEFAULT_N_CHANNELS,
    )


def _validated_simulation_options(o: SimulationRunOptions) -> SimulationRunOptions:
    """``o`` with its choices normalised; a usage error for anything invalid.

    The choices themselves are checked by click (MADIV-28 S2.2).
    """
    o = replace(
        o,
        source_type=o.source_type.lower(),
        save_mode=o.save_mode.lower(),
        backend=o.backend.lower(),
        imaging_algorithm=o.imaging_algorithm.lower(),
        background_mode=o.background_mode.lower(),
    )
    if not o.metadata_path.exists():
        usage_error(f"--metadata-path not found: {o.metadata_path}")
    o = _validated_input(o)
    if o.kaggle_token is not None and not o.kaggle_token.expanduser().exists():
        usage_error(f"--kaggle-token not found: {o.kaggle_token}")
    # 0 is the old default, which earlier runs pass explicitly.
    if o.background_level and o.background_mode == "none":
        usage_error("--background-level scales a background: choose one with --background-mode")
    try:
        model = resolve_noise_model(o.noise_model)
        if model is not None:
            model = model.select(
                only=[o.noise_only] if o.noise_only is not None else None,
                without=[o.noise_without] if o.noise_without else None,
            )
    except (ValueError, TypeError) as exc:  # json.JSONDecodeError and pydantic's are ValueErrors
        usage_error(f"--noise-model: {exc}")
    if model is None and (o.noise_only or o.noise_without):
        usage_error("--noise-only and --noise-without select components of a --noise-model")
    return replace(o, noise_model=model)


def _run_simulation_stages(
    o: SimulationRunOptions, params: SimulationParams, compute_backend: Any, *, label: str
) -> None:
    """The clean cube, observation, imaging and export of one run, with a progress bar."""
    with tqdm(
        total=100,
        desc=label,
        unit="%",
        leave=True,
    ) as progress_bar:
        logger = _progress_logger(progress_bar)
        is_sync_backend = o.backend == "sync"

        clean_stage = _run_stage_with_feedback(
            progress_bar,
            label="clean-cube",
            smooth_target=(34 if is_sync_backend else None),
            work=lambda: generate_clean_cube(
                params,
                logger=logger,
                compute_backend=compute_backend,
                progress_emitter=_progress_callback(progress_bar, 0, 35),
            ),
        )
        _update_progress(progress_bar, 35)
        logger(f"Clean cube shape: {clean_stage.model_cube.shape}")
        logger(f"Target ML shard: {params.ml_dataset_path}")

        simulation_results = _run_stage_with_feedback(
            progress_bar,
            label="observation",
            smooth_target=(89 if is_sync_backend else None),
            work=lambda: simulate_observation(
                clean_stage,
                compute_backend=compute_backend,
                robust=o.robust,
                terminal_logger=logger,
                interferometer_progress_callback=_progress_callback(progress_bar, 35, 55),
            ),
        )
        logger(f"Dirty cube shape: {simulation_results['dirty_cube'].shape}")
        logger(f"UV mask cube shape: {simulation_results['uv_mask_cube'].shape}")

        # The image-domain products (interferometric, total-power and
        # combined image cubes), as run_simulation makes them.
        simulation_results = _run_stage_with_feedback(
            progress_bar,
            label="imaging",
            work=lambda: image_products(simulation_results),
        )
        _update_progress(progress_bar, 90)

        exported_results = _run_stage_with_feedback(
            progress_bar,
            label="export",
            work=lambda: export_results(
                params,
                clean_stage,
                simulation_results,
                logger=logger,
            ),
        )
        _update_progress(progress_bar, 100)
    typer.echo(f"ML shard written to: {exported_results.get('ml_dataset_path')}")
    if o.save_mode != "memory":
        for key in sorted(exported_results):
            if key in {"ml_dataset_path", "antenna_config_paths"}:
                continue
            if key.endswith("_path"):
                typer.echo(f"{key}: {exported_results[key]}")
            elif key.endswith("_paths"):
                for path in exported_results[key]:
                    typer.echo(f"{key}: {path}")
