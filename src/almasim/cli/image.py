"""Batch imaging commands for the ALMASim CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from almasim.services.compute.progress import (
    TaskFailure,
    report_failures,
)
from almasim.services.imaging.batch_runner import (
    ImagingRunAborted,
    WorkersUnavailableError,
    run_imaging_tasks,
)
from almasim.services.imaging.ms_geometry import (  # noqa: F401  (moved; re-exported)
    MIN_IMAGE_PIXELS,
    SPECTRAL_COLUMNS,
    MissingDependencyError,
    _field_scan,
    add_spectral_columns,
    compute_imaging_parameters,
    image_pixels,
    import_casacore_tables,
    science_selection,
    spectral_window_properties,
)
from almasim.services.imaging.options import ExecOptions, ImagingOptions, SelfcalOptions
from almasim.services.imaging.wsclean_plan import (  # noqa: F401  (moved; re-exported)
    CUBE_PRODUCTS,
    CubeOptions,
    ImagingTask,
    _field_arguments,
    build_imaging_tasks,
    cube_channels,
    cube_plan,
    imaging_parameter_to_command_arg,
    row_dish_diameter,
    task_directory,
)
from almasim.services.instrument.constants import DEFAULT_MAX_WEIGHT
from almasim.services.ms.tables import (
    open_table,
    polarization_of_data_description,
    spw_of_data_description,
)

from ._options import (
    Backend,
    MaxCoresPerNode,
    PostprocessBackend,
    SchedulerHost,
    SchedulerInterface,
    SlurmMemory,
    SlurmProject,
    SlurmQueue,
    SlurmWalltime,
    TaskTimeout,
)
from ._shared import fail

image_app = typer.Typer(
    help="Visibility-to-image and image-to-visibility commands.",
    no_args_is_help=True,
)


def _iter_ms_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir() and input_path.name.endswith(".ms"):
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*.ms") if path.is_dir())
    raise typer.BadParameter(f"Input path is not a MeasurementSet or MS folder: {input_path}")


def _discover_models_for_ms(input_ms: Path, output_directory: Path) -> list[Path]:
    model_root = output_directory / input_ms.stem
    return sorted(model_root.glob("SPW-*/wsclean-model.fits"))


def _predict_all_models_for_ms(input_ms: Path, output_directory: Path) -> list[Path]:
    model_paths = _discover_models_for_ms(input_ms, output_directory)
    output_mss: list[Path] = []
    if not model_paths:
        typer.echo(
            f"[debug] missing model FITS, skipping MS: {input_ms}",
            err=False,
        )
        return output_mss

    for model_path in model_paths:
        output_ms = model_path.parent / f"{input_ms.name}.predicted"
        predict_from_model(input_ms=input_ms, model=model_path, output_ms=output_ms)
        output_mss.append(output_ms)

    return output_mss


def _run_commands_with_slurm_cluster(
    commands: list[tuple[str, list[str]]],
    *,
    cores_per_task: int,
    node_cores: int,
    queue: str,
    project: str | None,
    walltime: str,
    memory: str,
    n_jobs: int,
    scheduler_host: str | None,
    scheduler_interface: str | None,
    task_timeout: float | None,
) -> None:
    from almasim.services.compute import create_backend

    if not commands:
        return

    with create_backend(
        "slurm",
        queue=queue,
        node_cores=node_cores,
        memory=memory,
        walltime=walltime,
        n_workers=n_jobs,
        project=project,
        scheduler_host=scheduler_host,
        scheduler_interface=scheduler_interface,
    ) as backend:
        futures: list[tuple[str, Any]] = []
        for label, cmd in commands:
            future = backend.submit_subcommand(
                command=cmd,
                cores=cores_per_task,
                timeout=task_timeout,
            )
            futures.append((label, future))

        for label, future in tqdm(futures, total=len(futures), desc="SLURM tasks"):
            result = future.result()
            if result.returncode != 0:
                typer.echo(f"Task failed: {label}", err=True)
                typer.echo(result.stderr.rstrip(), err=True)
                raise typer.Exit(code=result.returncode)


@image_app.command("ms-overview", hidden=True)
def derive_parameters(
    input_ms: Path = typer.Argument(
        ...,
        help="Source of the measurement set",
    ),
):
    """Print the spectral-window summary for one MeasurementSet."""
    try:
        derived_parameters = compute_imaging_parameters(input_ms)
    except MissingDependencyError as exc:
        fail(str(exc))

    typer.echo(derived_parameters.to_string(index=False))


@image_app.command("compute-parameters")
def compute_parameters(
    archive_folder: Path = typer.Argument(
        default=Path("."),
        help="Processed MSs folder (default: current directory)",
    ),
    output_metadata_file: Path = typer.Argument(
        default=Path("imaging_parameters.csv"),
        help="Parameters CSV file (default: imaging_parameters.csv)",
    ),
    pattern: str = typer.Option(
        "*.cal",
        "--pattern",
        help="Glob for the MeasurementSets inside the folder.",
    ),
    science_only: bool = typer.Option(
        True,
        "--science-only/--all-spws",
        help=(
            "One row per spectral window and per field observed with the selected intent "
            "(OBSERVE_TARGET by default). --all-spws gives a row per field, calibrators "
            "included. Only (window, field) pairs with visibilities are listed."
        ),
    ),
    intent: list[str] | None = typer.Option(
        None,
        "--intent",
        help=(
            "Scan intent selecting the fields, e.g. CALIBRATE_PHASE#ON_SOURCE, or "
            "CALIBRATE_PHASE for any sub-intent. Repeat for several. Default: "
            "OBSERVE_TARGET. Not with --all-spws."
        ),
    ),
    require_done_marker: bool = typer.Option(
        True,
        "--require-done-marker/--no-require-done-marker",
        help=(
            "Skip an MS whose <ms>.done calibration marker is missing. Only applies when "
            "the folder holds at least one such marker, so plain folders still work."
        ),
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Abort on the first MS that cannot be read (default: record it and continue).",
    ),
):
    """Derive per-spectral-window imaging geometry for every MS in a folder.

    An MS that cannot be read (a truncated table, say) is reported and listed
    in ``<output>.failed.tsv`` instead of aborting the whole scan; the command
    then exits with status 1 once every MS has been attempted.
    """
    if intent and not science_only:
        raise typer.BadParameter("--intent selects fields; it cannot be combined with --all-spws")
    mss = sorted(archive_folder.glob(pattern))
    if len(mss) == 0:
        typer.echo(f"Cannot find any MS in {archive_folder}")
        raise typer.Exit(code=1)

    if require_done_marker:
        with_marker = [ms for ms in mss if ms.with_name(ms.name + ".done").is_file()]
        if with_marker and len(with_marker) < len(mss):
            typer.echo(
                f"Skipping {len(mss) - len(with_marker)} MS(s) without a .done calibration marker."
            )
            mss = with_marker

    frames: list[pd.DataFrame] = []
    failures: list[tuple[Path, str]] = []
    for ms in tqdm(mss):
        try:
            frames.append(compute_imaging_parameters(ms, science_only=science_only, intents=intent))
        except Exception as exc:
            if fail_fast:
                raise
            failures.append((ms, f"{type(exc).__name__}: {exc}"))
            typer.echo(f"FAILED {ms.name}: {type(exc).__name__}: {exc}", err=True)

    if frames:
        main_output = pd.concat(frames, axis=0, ignore_index=True)
        main_output.to_csv(output_metadata_file, index=False)
        typer.echo(
            f"Wrote {len(main_output)} (spectral window, field) row(s) for {len(frames)} MS(s) "
            f"to {output_metadata_file}"
        )
    if failures:
        failed_path = output_metadata_file.with_name(output_metadata_file.name + ".failed.tsv")
        failed_path.write_text(
            "".join(f"{ms}\t{error}\n" for ms, error in failures), encoding="utf-8"
        )
        fail(f"{len(failures)} MS(s) could not be read; listed in {failed_path}")


def _echo_cube_plan(
    parameters: pd.DataFrame,
    fov_fraction: float,
    beam_sampling: float,
    options: CubeOptions,
    products: Sequence[str],
    min_beams_across: float = 0.0,
) -> None:
    plan = cube_plan(parameters, fov_fraction, beam_sampling, options, min_beams_across)
    if plan.empty:
        return
    averaged = int((plan["cube_channels"] < plan["n_channels"]).sum())
    capped = plan["size_limited"]
    typer.echo(
        f"Cube plan for {len(plan)} task(s): channels per cube "
        f"min {plan['cube_channels'].min()}, median {int(plan['cube_channels'].median())}, "
        f"max {plan['cube_channels'].max()}; {averaged} window(s) averaged below their "
        f"native channels; {int(capped.sum())} given fewer channels by the "
        f"{options.max_cube_gb:g} GB cap."
    )
    typer.echo(
        f"Projected volume: {plan['cube_gb'].sum() / 1e3:.1f} TB per product, "
        f"{len(products) * plan['cube_gb'].sum() / 1e3:.1f} TB for {', '.join(products)}."
    )


def _report_imaging_failures(failures: list[TaskFailure]) -> None:
    report_failures("Image", failures, echo=lambda t: typer.echo(t, err=True))


def _prepare_selfcal_casa_data(output_directory: Path, casa_data_root: Path | None) -> Path:
    """CASA runtime data for the workers' gaincal, populated here once if missing.

    The calibration stage's preflight: the tasks would otherwise each try to
    download it, or fail to import casatools without it.
    """
    from ..services.casa.environment import configure_casa_environment
    from ..services.casa.runtime_data import ensure_casa_runtime_data

    output_directory = Path(output_directory).expanduser().absolute()
    casa_data = (
        Path(casa_data_root).expanduser().absolute()
        if casa_data_root is not None
        else output_directory / ".casa-data"
    )
    workspace = output_directory / ".casa-work"
    configure_casa_environment(workspace, casa_data, workspace_root=workspace)
    typer.echo(f"Self-calibration CASA runtime data: {casa_data}")
    ensure_casa_runtime_data(casa_data, logger_fn=typer.echo)
    return casa_data


@dataclass(frozen=True)
class ImageFromMsOptions:
    """Every option of ``image image-from-ms`` (MADIV-28 T3.8)."""

    imaging_parameters: Path
    output_directory: Path
    fov_fraction: float
    beam_sampling: float
    min_beams_across: float
    multiscale: bool
    num_cores: int
    max_cores_per_node: MaxCoresPerNode
    wsclean_bin: str
    auto_threshold: float
    auto_mask: float
    task_memory_gb: float
    single_window: bool
    task_retries: int
    max_weight: float
    scratch_dir: Path | None
    continuum_restart: bool
    update_model: bool
    postprocess_backend: PostprocessBackend
    slurm_queue: SlurmQueue
    slurm_project: SlurmProject
    slurm_walltime: SlurmWalltime
    slurm_memory: SlurmMemory
    slurm_n_jobs: int
    scheduler_host: SchedulerHost
    scheduler_interface: SchedulerInterface
    task_timeout: TaskTimeout
    skip_existing: bool
    overwrite_outputs: bool
    continue_on_error: bool
    continuum: bool
    cube: bool
    contsub: bool
    selfcal: str
    selfcal_min_snr: float
    noise_products: bool
    pbcor: bool
    casa_data_root: Path | None
    uv_contsub: bool
    contsub_order: int
    spectral_sampling: float
    velocity_resolution: float
    max_cube_gb: float
    cube_products: str
    join_channels: bool
    cube_line_mask: bool
    cube_reorder: bool
    plan_only: bool
    ous_index: Path | None


@image_app.command("image-from-ms")
def image_from_ms(
    imaging_parameters: Path = typer.Argument(help="Imaging parameter file"),
    output_directory: Path = typer.Argument(help="Output directory path"),
    fov_fraction: float = typer.Option(
        help="Fraction of the FOV to image",
        default=1.5,
        min=1e-6,
        rich_help_panel="Imaging",
    ),
    beam_sampling: float = typer.Option(
        help="Number of pixels to use to sample the synthetized beam. Could be fractional.",
        default=8,
        min=1e-6,
        rich_help_panel="Imaging",
    ),
    # Compact-configuration images were ~12 beams wide: the source's sidelobes
    # filled them, inflated the noise WSClean cleaned to and left more source
    # in the residual. At 32 beams the line-free residual share of four
    # cycle11 cubes fell from 7-25 % to 6-21 % and their S/N rose 1.4-4x, for
    # ~13 % more run time; multi-scale changed nothing on these compact
    # sources and diverged on a 6.6 Jy one in a 94 px image (2026-10-17).
    min_beams_across: float = typer.Option(
        32.0,
        "--min-beams-across",
        min=0.0,
        help=(
            "Make every image at least this many synthesized beams wide, whatever "
            "--fov-fraction gives (default 32; 0: no minimum)."
        ),
        rich_help_panel="Imaging",
    ),
    multiscale: bool = typer.Option(
        False,
        "--multiscale/--no-multiscale",
        help=(
            "Clean with WSClean's multi-scale algorithm (-multiscale) for extended emission. "
            "Off by default: it can diverge on bright sources in small images."
        ),
        rich_help_panel="Imaging",
    ),
    num_cores: int = typer.Option(
        help="Number of cores per imaging task", default=10, min=1, rich_help_panel="Cluster"
    ),
    max_cores_per_node: MaxCoresPerNode = 95,
    wsclean_bin: str = typer.Option(
        "wsclean",
        "--wsclean-bin",
        help="Path or executable name of the WSClean binary the workers run.",
        rich_help_panel="Advanced",
    ),
    auto_threshold: float = typer.Option(
        1.0,
        "--auto-threshold",
        min=0.0,
        help=(
            "Stop cleaning when the residual peak drops below this many sigma of the "
            "residual noise (WSClean -auto-threshold; default 1, inside the --auto-mask "
            "mask). 0 disables it and cleaning runs to -niter."
        ),
        rich_help_panel="Imaging",
    ),
    auto_mask: float = typer.Option(
        3.0,
        "--auto-mask",
        min=0.0,
        help=(
            "WSClean -auto-mask level in sigma (default 3): clean to this level everywhere, "
            "then only inside the mask it found down to --auto-threshold. Must be above "
            "--auto-threshold; 0 disables masking."
        ),
        rich_help_panel="Imaging",
    ),
    task_memory_gb: float = typer.Option(
        0.0,
        "--task-memory-gb",
        min=0.0,
        help=(
            "Absolute memory limit per WSClean task in GB (WSClean -abs-mem). Use the "
            "worker's Slurm allocation divided by the tasks per node. 0 (default) falls back "
            "to -mem <num-cores/max-cores-per-node> percent of the node."
        ),
        rich_help_panel="Cluster",
    ),
    single_window: bool = typer.Option(
        True,
        "--single-window-ms/--no-single-window-ms",
        help=(
            "Extract each task's spectral window (and target fields) into a per-task MS on "
            "the worker and image it with -no-reorder (default). WSClean 3.7's reordering "
            "reads out of bounds on multi-window ALMA sets and crashes about half the tasks; "
            "--no-single-window-ms images the original MS with -spws/-field instead."
        ),
        rich_help_panel="Advanced",
    ),
    task_retries: int = typer.Option(
        3,
        "--task-retries",
        min=0,
        help=(
            "Retry a task whose WSClean died by a signal (segfault) this many times before "
            "marking it failed. WSClean 3.7's multi-threaded reordering crashes "
            "non-deterministically on some inputs; a crash costs seconds."
        ),
        rich_help_panel="Advanced",
    ),
    max_weight: float = typer.Option(
        DEFAULT_MAX_WEIGHT,
        "--max-weight",
        min=0.0,
        help=(
            "Leave visibility rows whose WEIGHT exceeds this out of the single-window MS "
            "(0 keeps every row). Calibrated ALMA rows the pipeline had flagged but that "
            "were never re-flagged carry WEIGHT ~1e5-1e8 with garbage amplitudes and would "
            "dominate the image; physical weights stay below ~1e3."
        ),
        rich_help_panel="Advanced",
    ),
    scratch_dir: Path | None = typer.Option(
        None,
        "--scratch-dir",
        help=(
            "Directory for WSClean's reordered visibilities (one sub-directory per task, "
            "removed afterwards). Default: the task's output directory."
        ),
        rich_help_panel="Advanced",
    ),
    continuum_restart: bool = typer.Option(
        True,
        "--continuum-restart/--no-continuum-restart",
        help=(
            "Clean the continuum in two passes, the second (-continue) from the true residual "
            "with a mask rebuilt at the true noise: the first pass's auto-mask, built while "
            "a bright source's sidelobes inflate the noise, misses its fainter parts "
            "(default; only with --single-window-ms, whose per-task copy gets MODEL_DATA)."
        ),
        rich_help_panel="Imaging",
    ),
    update_model: bool = typer.Option(
        False,
        "--update-model/--no-update-model",
        help=(
            "Write the CLEAN model back into the MS MODEL_DATA column (WSClean "
            "-update-model-required). Off by default: concurrent spectral-window tasks "
            "would write the same MS and every calibrated MS would grow."
        ),
        rich_help_panel="Advanced",
    ),
    postprocess_backend: PostprocessBackend = Backend.slurm,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "02:00:00",
    slurm_memory: SlurmMemory = "16GB",
    slurm_n_jobs: int = typer.Option(
        1,
        "--slurm-workers",
        "--slurm-n-jobs",
        min=1,
        help="Number of Slurm workers (jobs).",
        rich_help_panel="Cluster",
    ),
    scheduler_host: SchedulerHost = None,
    scheduler_interface: SchedulerInterface = None,
    task_timeout: TaskTimeout = 3600,
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help=(
            "Also trust an existing wsclean-image.fits that has no .done marker (an output "
            "from before markers existed) and write its marker. Tasks with a marker are "
            "always skipped."
        ),
        rich_help_panel="Outputs",
    ),
    overwrite_outputs: bool = typer.Option(
        False,
        "--overwrite-outputs",
        help="Redo every task, including the ones with a .done marker.",
        rich_help_panel="Outputs",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help=(
            "Keep imaging the rest of the batch when a task fails (default); the failed "
            "tasks are listed at the end and the command exits 1. --fail-fast stops at the "
            "first failure."
        ),
        rich_help_panel="Outputs",
    ),
    continuum: bool = typer.Option(
        True,
        "--continuum/--no-continuum",
        help=(
            "Make the continuum image of every window (wsclean-image.fits; default). "
            "--no-continuum with --cube adds cubes next to existing continuum images."
        ),
        rich_help_panel="Imaging",
    ),
    cube: bool = typer.Option(
        False,
        "--cube/--no-cube",
        help=(
            "Also image every window as a spectral cube (WSClean -channels-out, the channel "
            "count chosen per window by --spectral-sampling, --velocity-resolution and "
            "--max-cube-gb), stacked into wsclean-cube-<product>.fits."
        ),
        rich_help_panel="Spectral cube",
    ),
    contsub: bool = typer.Option(
        False,
        "--contsub/--no-contsub",
        help=(
            "Also write the image cube continuum-subtracted (wsclean-cube-image.contsub.fits): "
            "a polynomial fitted along frequency per pixel, line channels found by 3-sigma "
            "clipping. Off by default. Needs --cube."
        ),
        rich_help_panel="Continuum subtraction",
    ),
    selfcal: str = typer.Option(
        "",
        "--selfcal",
        help=(
            "Self-calibrate each window with CASA gaincal before imaging, one round per "
            "comma-separated solution interval: inf (per scan), int (per integration) or "
            "<seconds>s for phase; ap:<interval> for amplitude and phase on top of the "
            "phase solution, e.g. 'inf,30s,ap:inf'. A round is kept only if the continuum "
            "S/N rises. Empty (default): off. Needs --continuum and --single-window-ms."
        ),
        rich_help_panel="Self-calibration",
    ),
    selfcal_min_snr: float = typer.Option(
        20.0,
        "--selfcal-min-snr",
        min=0.0,
        help="Skip self-calibration when the initial continuum S/N is below this.",
        rich_help_panel="Self-calibration",
    ),
    noise_products: bool = typer.Option(
        True,
        "--noise-products/--no-noise-products",
        help=(
            "Where self-calibration was applied, also image what it removed and keep it: "
            "wsclean[-cube]-selfcal-correction.fits, wsclean[-cube]-residual-total.fits "
            "(residual + correction: all the data the model does not explain) and the gain "
            "tables in selfcal_gains/ (default)."
        ),
        rich_help_panel="Quality products",
    ),
    pbcor: bool = typer.Option(
        False,
        "--pbcor/--no-pbcor",
        help=(
            "Also primary-beam correct the continuum image and the image cube: "
            "wsclean[-cube]-pb.fits (the beam, CASA's ALMA Airy model, per channel) and "
            "wsclean[-cube]-image.pbcor.fits (blanked below 0.2). Off by default; "
            "'image pb-correct' adds them to an existing run."
        ),
        rich_help_panel="Quality products",
    ),
    casa_data_root: Path | None = typer.Option(
        None,
        "--casa-data-root",
        help=(
            "CASA runtime data for --selfcal (default <output>/.casa-data, downloaded once "
            "by the driver when missing, as the calibration stage does)."
        ),
        rich_help_panel="Self-calibration",
    ),
    uv_contsub: bool = typer.Option(
        False,
        "--uv-contsub/--no-uv-contsub",
        help=(
            "Subtract the continuum from the visibilities before imaging (CASA uvcontsub): "
            "line channels are found on the source spectrum, the continuum is fitted per "
            "visibility over the others. The continuum image is made from the fitted "
            "continuum and every cube product holds only the lines. Needs --cube and "
            "--single-window-ms; replaces --contsub."
        ),
        rich_help_panel="Continuum subtraction",
    ),
    contsub_order: int = typer.Option(
        1,
        "--contsub-order",
        min=0,
        max=1,
        help=(
            "Continuum polynomial order for --contsub and --uv-contsub: 0 (constant) or 1 "
            "(linear, default)."
        ),
        rich_help_panel="Continuum subtraction",
    ),
    spectral_sampling: float = typer.Option(
        2.0,
        "--spectral-sampling",
        min=1e-6,
        help=(
            "Cube channels per spectral resolution element (the spectral counterpart of "
            "--beam-sampling). 2 keeps the native channels of ALMA's Hanning-smoothed "
            "windows; 1 halves them."
        ),
        rich_help_panel="Spectral cube",
    ),
    velocity_resolution: float = typer.Option(
        0.0,
        "--velocity-resolution",
        min=0.0,
        help="Narrowest cube channel in km/s at the window's reference frequency (0: no limit).",
        rich_help_panel="Spectral cube",
    ),
    max_cube_gb: float = typer.Option(
        4.0,
        "--max-cube-gb",
        min=1e-3,
        help=(
            "Largest cube per product in GB (32-bit pixels): windows imaged on large grids "
            "get fewer, wider channels. Channel counts are divisors of the native count, so "
            "every cube channel averages the same number of native channels."
        ),
        rich_help_panel="Spectral cube",
    ),
    cube_products: str = typer.Option(
        ",".join(CUBE_PRODUCTS),
        "--cube-products",
        help=(
            "Comma-separated cube products to keep: image, residual, psf, model, dirty "
            "(default: all five). The others are removed (dirty is not even made)."
        ),
        rich_help_panel="Spectral cube",
    ),
    # On uid___A002_X10ded83_X6af4 SPW 16 joining cut the stacked line-free
    # residual from 12 % to 8 % of the image but raised the line channels'
    # from 11 % to 30 % (2026-10-17), so it stays opt-in.
    join_channels: bool = typer.Option(
        False,
        "--join-channels/--no-join-channels",
        help=(
            "Clean the cube on the sum of its channels (WSClean -join-channels): "
            "continuum too faint to clean channel by channel is cleaned too, but narrow "
            "lines are cleaned worse. Off by default. Needs --cube."
        ),
        rich_help_panel="Spectral cube",
    ),
    cube_line_mask: bool = typer.Option(
        True,
        "--cube-line-mask/--no-cube-line-mask",
        help=(
            "Clean the cube a second time (-continue) inside the moment-0 mask of the first "
            "pass: a line's faint parts, below each channel's own auto-mask but clear in the "
            "sum, are cleaned too (default; with --cube and --single-window-ms; ~2x the cube "
            "time)."
        ),
        rich_help_panel="Spectral cube",
    ),
    cube_reorder: bool = typer.Option(
        True,
        "--cube-reorder/--cube-no-reorder",
        help=(
            "Let WSClean reorder the single-window MS per channel for the cube (default). "
            "Without it WSClean re-reads the whole MS for every cube channel: ~25x slower."
        ),
        rich_help_panel="Advanced",
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Print the channel plan and the projected cube volume, then exit without imaging.",
        rich_help_panel="Spectral cube",
    ),
    ous_index: Path | None = typer.Option(
        None,
        "--ous-index",
        help=(
            "The delivery tree's ous_index.csv, for the project, science goal, group and "
            "member OUS in every FITS header (default: the ous_index.csv nearest above each MS)."
        ),
        rich_help_panel="Imaging",
    ),
):
    """Image every (MS, spectral window) row of a parameter CSV with WSClean.

    Each task leaves ``SPW-<n>.done`` or ``SPW-<n>.failed`` (with the error and
    the log path) next to its output directory, and ``SPW-<n>.log`` with the
    full WSClean output. A rerun skips the tasks with a ``.done`` marker and
    retries the rest, so an interrupted batch can simply be resubmitted.
    """
    o = ImageFromMsOptions(**locals())
    imaging, execution = _image_from_ms_options(o)
    tasks = _image_from_ms_tasks(o, imaging)
    if tasks:
        _run_image_from_ms(o, imaging, execution, tasks)


def _image_from_ms_options(o: ImageFromMsOptions) -> tuple[ImagingOptions, ExecOptions]:
    """What every task makes and where the tasks run, validated."""
    # A 3-sigma stop without a mask left ~3 sigma of source in every channel's
    # residual: stacked over the 3537 line-free channels of uid___A002_X10ded83_X6af4
    # SPW 16 that was 7.9 sigma and 39 % of the image; masking to 3 and cleaning
    # to 1 sigma left 4.1 sigma and 16 % (2026-10-17).
    if 0 < o.auto_mask <= o.auto_threshold:
        raise typer.BadParameter("--auto-mask must be above --auto-threshold (or 0 to disable it)")
    if o.plan_only and not o.cube:
        raise typer.BadParameter(
            "Options that need --cube: --plan-only, --contsub, --uv-contsub, --join-channels"
        )
    imaging = ImagingOptions(
        continuum=o.continuum,
        cube=(
            CubeOptions(
                spectral_sampling=o.spectral_sampling,
                velocity_resolution_km_s=o.velocity_resolution,
                max_cube_gb=o.max_cube_gb,
            )
            if o.cube
            else None
        ),
        cube_products=(
            tuple(p.strip().lower() for p in o.cube_products.split(",") if p.strip())
            if o.cube
            else None
        ),
        single_window=o.single_window,
        max_weight=o.max_weight if o.max_weight > 0 else None,
        contsub=o.contsub,
        uv_contsub=o.uv_contsub,
        contsub_order=o.contsub_order,
        join_channels=o.join_channels,
        continuum_restart=o.continuum_restart,
        cube_line_mask=o.cube_line_mask,
        pbcor=o.pbcor,
        selfcal=SelfcalOptions(
            solints=tuple(part.strip() for part in o.selfcal.split(",") if part.strip()),
            min_snr=o.selfcal_min_snr,
            noise_products=o.noise_products,
        ),
    )
    execution = ExecOptions(
        backend=o.postprocess_backend.lower(),
        cores_per_task=o.num_cores,
        node_cores=o.max_cores_per_node,
        queue=o.slurm_queue,
        project=o.slurm_project,
        walltime=o.slurm_walltime,
        memory=o.slurm_memory,
        n_jobs=o.slurm_n_jobs,
        scheduler_host=o.scheduler_host,
        scheduler_interface=o.scheduler_interface,
        task_timeout=o.task_timeout,
        task_retries=o.task_retries,
        continue_on_error=o.continue_on_error,
        scratch_root=str(o.scratch_dir) if o.scratch_dir is not None else None,
    )
    try:
        imaging.validate()
        execution.validate()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return imaging, execution


def _image_from_ms_tasks(o: ImageFromMsOptions, imaging: ImagingOptions) -> list[ImagingTask]:
    """The tasks still to image (none with --plan-only), after the cube plan is shown."""
    parameters = pd.read_csv(str(o.imaging_parameters))
    if imaging.cube is not None:
        parameters = add_spectral_columns(parameters)
        _echo_cube_plan(
            parameters,
            o.fov_fraction,
            o.beam_sampling,
            imaging.cube,
            imaging.cube_products or (),
            o.min_beams_across,
        )
    if o.plan_only:
        return []
    tasks, skipped = build_imaging_tasks(
        parameters,
        o.output_directory,
        fov_fraction=o.fov_fraction,
        beam_sampling=o.beam_sampling,
        num_cores=o.num_cores,
        max_cores_per_node=o.max_cores_per_node,
        wsclean_bin=o.wsclean_bin,
        overwrite_outputs=o.overwrite_outputs,
        trust_existing_images=o.skip_existing,
        auto_threshold=o.auto_threshold if o.auto_threshold > 0 else None,
        auto_mask=o.auto_mask if o.auto_mask > 0 else None,
        update_model=o.update_model,
        task_memory_gb=o.task_memory_gb if o.task_memory_gb > 0 else None,
        scratch_dir=str(o.scratch_dir) if o.scratch_dir is not None else None,
        single_window=imaging.single_window,
        continuum=imaging.continuum,
        cube=imaging.cube,
        cube_products=imaging.cube_products,
        cube_reorder=o.cube_reorder,
        min_beams_across=o.min_beams_across,
        multiscale=o.multiscale,
        join_channels=o.join_channels,
        ous_index_csv=o.ous_index,
    )
    if skipped:
        typer.echo(f"Skipped {skipped} already-imaged SPW(s).")
    if not tasks:
        typer.echo("All requested tasks are already imaged; nothing to do.")
        return []
    return tasks


def _run_image_from_ms(
    o: ImageFromMsOptions,
    imaging: ImagingOptions,
    execution: ExecOptions,
    tasks: list[ImagingTask],
) -> None:
    """Run the tasks and report them; exit 1 when any failed."""
    casa_data: str | None = None
    if imaging.selfcal.solints:
        casa_data = str(_prepare_selfcal_casa_data(o.output_directory, o.casa_data_root))
    typer.echo(
        f"Imaging {len(tasks)} (MS, SPW, field) task(s); "
        "per-task logs: <output>/<ms>/SPW-<n>[_FIELD-<id>].log"
    )

    failures: list[TaskFailure] = []
    try:
        run_imaging_tasks(
            tasks,
            imaging,
            execution,
            casa_data=casa_data,
            failures=failures,
            echo=typer.echo,
        )
    except ImagingRunAborted as exc:
        raise typer.Exit(code=1) from exc
    except WorkersUnavailableError as exc:
        fail(str(exc))
    typer.echo(f"Imaged {len(tasks) - len(failures)}/{len(tasks)} task(s), {len(failures)} failed.")
    if failures:
        _report_imaging_failures(failures)
        raise typer.Exit(code=1)


@image_app.command("clean-quality")
def clean_quality(
    output_root: Path = typer.Argument(
        help="An image-from-ms output directory (or one task directory) to assess."
    ),
    moments: bool = typer.Option(
        True,
        "--moments/--no-moments",
        help="Write the cube moment maps (quality/cube-*.fits) next to each task's images.",
    ),
    summary: Path | None = typer.Option(
        None,
        "--summary",
        help="Summary CSV path (default <output_root>/clean-quality.csv).",
    ),
):
    """Measure how much source signal the clean left in the residuals, task by task.

    For every task under OUTPUT_ROOT writes quality.json (continuum and cube
    statistics: S/N, the residual's share and profile at the source, the
    noise WSClean measured when it built its mask, line and line-free
    residuals, model fluxes) and, for cubes, moment maps of the line and of
    the residual; prints one row per task and writes them to a CSV. See
    almasim.services.imaging.quality for each statistic.
    """
    from ..services.imaging.quality import (
        SUMMARY_FILENAME,
        assess_task,
        find_task_dirs,
        summary_rows,
        write_summary,
    )

    root = Path(output_root).expanduser().absolute()
    tasks = find_task_dirs(root)
    if not tasks:
        typer.echo(f"No imaged tasks (wsclean-image.fits or wsclean-cube-image.fits) under {root}")
        raise typer.Exit(code=1)
    results = []
    for task in tqdm(tasks, desc="clean quality", unit="task"):
        try:
            results.append(assess_task(task, write_moments=moments))
        except Exception as exc:  # noqa: BLE001 - one broken task must not hide the others
            typer.echo(f"  {task}: {type(exc).__name__}: {exc}", err=True)
    if not results:
        fail(f"None of the {len(tasks)} task(s) under {root} could be assessed")
    rows = summary_rows(results, root)
    summary_path = summary or root / SUMMARY_FILENAME
    write_summary(rows, summary_path)
    frame = pd.DataFrame(rows).set_index("task")
    frame.columns = [column.replace("continuum.", "cont.") for column in frame.columns]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        typer.echo(frame.to_string(float_format=lambda value: f"{value:.3g}"))
    typer.echo(f"Wrote {len(results)} quality.json file(s) and {summary_path}")


@image_app.command("pb-correct")
def pb_correct(
    imaging_parameters: Path = typer.Argument(help="Parameter CSV the outputs were imaged from"),
    output_directory: Path = typer.Argument(help="The image-from-ms output directory"),
    cutoff: float = typer.Option(
        0.2,
        "--cutoff",
        min=1e-3,
        max=0.999,
        help="Blank the corrected images where the beam's response is below this.",
    ),
):
    """Primary-beam correct the images of an existing image-from-ms run.

    For every row of IMAGING_PARAMETERS whose task finished (a .done marker)
    writes wsclean[-cube]-pb.fits and wsclean[-cube]-image.pbcor.fits next to
    its continuum image and image cube, as image-from-ms --pbcor does, without
    imaging again. The dish diameter comes from the row's fov_per_frequency.
    """
    from ..services.imaging.markers import imaging_log_path, imaging_marker_path
    from ..services.imaging.task import pb_correct_task, pb_note

    parameters = pd.read_csv(str(imaging_parameters))
    root = Path(output_directory).expanduser().absolute()
    corrected = failed = skipped = 0
    for _, row in tqdm(list(parameters.iterrows()), desc="pb-correct", unit="task"):
        outdir, _ = task_directory(root, row)
        if not imaging_marker_path(outdir).is_file():
            skipped += 1
            continue
        dish_diameter = row_dish_diameter(row)
        try:
            if dish_diameter is None:
                raise ValueError("the row has no fov_per_frequency/reference_frequency")
            done = pb_correct_task(outdir, dish_diameter, cutoff)
        except Exception as exc:  # noqa: BLE001 - one broken task must not stop the others
            failed += 1
            typer.echo(f"  {outdir}: {type(exc).__name__}: {exc}", err=True)
            continue
        with imaging_log_path(outdir).open("a", encoding="utf-8") as log:
            for name, info in done.items():
                log.write(f"# primary-beam corrected {name} (pb-correct): {pb_note(info)}\n")
        corrected += 1
    typer.echo(
        f"Primary-beam corrected {corrected} task(s); {skipped} not imaged, {failed} failed."
    )
    if failed or not corrected:
        raise typer.Exit(code=1)


@image_app.command("batch-image", hidden=True)
def image_set(
    imaging_parameters: Path = typer.Argument(help="Imaging parameter file"),
    output_directory: Path = typer.Argument(help="Output directory path"),
    fov_fraction: float = typer.Option(
        help="Fraction of the FOV to image",
        default=1.5,
        min=1e-6,
        rich_help_panel="Imaging",
    ),
    beam_sampling: float = typer.Option(
        help="Number of pixels to use to sample the synthetized beam. Could be fractional.",
        default=8,
        min=1e-6,
        rich_help_panel="Imaging",
    ),
    num_cores: int = typer.Option(
        help="Number of cores per imaging task", default=10, min=1, rich_help_panel="Cluster"
    ),
    max_cores_per_node: MaxCoresPerNode = 95,
    wsclean_bin: str = typer.Option("wsclean", "--wsclean-bin", rich_help_panel="Advanced"),
    auto_threshold: float = typer.Option(
        1.0, "--auto-threshold", min=0.0, rich_help_panel="Imaging"
    ),
    auto_mask: float = typer.Option(3.0, "--auto-mask", min=0.0, rich_help_panel="Imaging"),
    update_model: bool = typer.Option(
        False, "--update-model/--no-update-model", rich_help_panel="Advanced"
    ),
    task_memory_gb: float = typer.Option(
        0.0, "--task-memory-gb", min=0.0, rich_help_panel="Cluster"
    ),
    task_retries: int = typer.Option(3, "--task-retries", min=0, rich_help_panel="Advanced"),
    single_window: bool = typer.Option(
        True, "--single-window-ms/--no-single-window-ms", rich_help_panel="Advanced"
    ),
    max_weight: float = typer.Option(10000.0, "--max-weight", min=0.0, rich_help_panel="Advanced"),
    scratch_dir: Path | None = typer.Option(None, "--scratch-dir", rich_help_panel="Advanced"),
    postprocess_backend: PostprocessBackend = Backend.slurm,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "02:00:00",
    slurm_memory: SlurmMemory = "16GB",
    slurm_n_jobs: int = typer.Option(
        1,
        "--slurm-workers",
        "--slurm-n-jobs",
        min=1,
        help="Number of Slurm workers (jobs).",
        rich_help_panel="Cluster",
    ),
    scheduler_host: SchedulerHost = None,
    scheduler_interface: SchedulerInterface = None,
    task_timeout: TaskTimeout = 3600,
    skip_existing: bool = typer.Option(False, "--skip-existing", rich_help_panel="Outputs"),
    overwrite_outputs: bool = typer.Option(False, "--overwrite-outputs", rich_help_panel="Outputs"),
    continue_on_error: bool = typer.Option(
        True, "--continue-on-error/--fail-fast", rich_help_panel="Outputs"
    ),
):
    """Alias of ``image-from-ms``."""
    image_from_ms(
        imaging_parameters=imaging_parameters,
        output_directory=output_directory,
        fov_fraction=fov_fraction,
        beam_sampling=beam_sampling,
        min_beams_across=32.0,
        multiscale=False,
        num_cores=num_cores,
        max_cores_per_node=max_cores_per_node,
        wsclean_bin=wsclean_bin,
        auto_threshold=auto_threshold,
        auto_mask=auto_mask,
        update_model=update_model,
        continuum_restart=True,
        task_memory_gb=task_memory_gb,
        task_retries=task_retries,
        single_window=single_window,
        max_weight=max_weight,
        scratch_dir=scratch_dir,
        continuum=True,
        cube=False,
        contsub=False,
        uv_contsub=False,
        selfcal="",
        selfcal_min_snr=20.0,
        noise_products=True,
        pbcor=False,
        casa_data_root=None,
        contsub_order=1,
        spectral_sampling=2.0,
        velocity_resolution=0.0,
        max_cube_gb=4.0,
        cube_products=",".join(CUBE_PRODUCTS),
        join_channels=False,
        cube_line_mask=True,
        cube_reorder=True,
        plan_only=False,
        ous_index=None,
        postprocess_backend=postprocess_backend,
        slurm_queue=slurm_queue,
        slurm_project=slurm_project,
        slurm_walltime=slurm_walltime,
        slurm_memory=slurm_memory,
        slurm_n_jobs=slurm_n_jobs,
        scheduler_host=scheduler_host,
        scheduler_interface=scheduler_interface,
        task_timeout=task_timeout,
        skip_existing=skip_existing,
        overwrite_outputs=overwrite_outputs,
        continue_on_error=continue_on_error,
    )


@image_app.command("ms-from-image", hidden=True)
def ms_from_image(
    input_ms_or_folder: Path = typer.Argument(help="Single MS directory or folder of MSs"),
    output_directory: Path = typer.Argument(help="Directory containing model FITS products"),
    use_slurm: bool = typer.Option(
        help="Whether or not to use slurm or not", default=True, rich_help_panel="Cluster"
    ),
    num_cores: int = typer.Option(
        help="Number of cores per predict task", default=1, min=1, rich_help_panel="Cluster"
    ),
    max_cores_per_node: MaxCoresPerNode = 95,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "02:00:00",
    slurm_memory: SlurmMemory = "16GB",
    slurm_n_jobs: int = typer.Option(
        1,
        "--slurm-workers",
        "--slurm-n-jobs",
        min=1,
        help="Number of Slurm workers (jobs).",
        rich_help_panel="Cluster",
    ),
    scheduler_host: SchedulerHost = None,
    scheduler_interface: SchedulerInterface = None,
    task_timeout: TaskTimeout = 3600,
):
    from .predict import ms_from_image as _predict_ms_from_image

    _predict_ms_from_image(
        input_ms_or_folder=input_ms_or_folder,
        output_directory=output_directory,
        use_slurm=use_slurm,
        num_cores=num_cores,
        max_cores_per_node=max_cores_per_node,
        slurm_queue=slurm_queue,
        slurm_project=slurm_project,
        slurm_walltime=slurm_walltime,
        slurm_memory=slurm_memory,
        slurm_n_jobs=slurm_n_jobs,
        scheduler_host=scheduler_host,
        scheduler_interface=scheduler_interface,
        task_timeout=task_timeout,
    )


@image_app.command("predict-single", hidden=True)
def predict_single(
    input_ms: Path = typer.Argument(..., help="Input MS"),
    model: Path = typer.Argument(..., help="FITS model path"),
    output_ms: Path = typer.Argument(..., help="Output MS"),
):
    predict_from_model(input_ms=input_ms, model=model, output_ms=output_ms)


# CORR_TYPE codes (casacore Stokes enum) of the parallel hands, where a Stokes I
# model appears: XX = YY = I and RR = LL = I, since I = (XX + YY) / 2.
_PARALLEL_HANDS = {5, 8, 9, 12}  # RR, LL, XX, YY


def _model_planes(model: Path) -> tuple[np.ndarray, float, float]:
    """Stokes I planes (nplane, ny, nx) of a FITS model, and its pixel sizes in radians."""
    from astropy.io import fits

    with fits.open(model) as hdul:
        hdu = cast(fits.PrimaryHDU, hdul[0])
        if hdu.data is None:
            raise ValueError(f"FITS primary HDU of {model} contains no image data")
        header = hdu.header
        data = np.asarray(hdu.data, dtype=np.float64)
    # FITS axes (numpy order) are [stokes, freq, y, x]; keep Stokes I.
    while data.ndim > 3:
        data = data[0]
    planes = data.reshape((-1,) + data.shape[-2:])
    deg2rad = np.pi / 180.0
    pixsize_x = abs(cast(float, header["CDELT1"])) * deg2rad
    pixsize_y = abs(cast(float, header["CDELT2"])) * deg2rad
    return np.ascontiguousarray(planes), pixsize_x, pixsize_y


def predict_from_model(input_ms: Path, model: Path, output_ms: Path) -> None:
    """Copy ``input_ms`` to ``output_ms`` and write the model's visibilities to MODEL_DATA.

    The visibilities are predicted with ducc0 (``interferometry.predict``) at
    each spectral window's channel frequencies (from the MS). A single-plane
    model (e.g. WSClean's MFS model) is used for every channel; a cube model
    must have one plane per channel and each channel is predicted from its
    own plane. The Stokes I model goes to the parallel hands (XX = YY = I, or
    RR = LL = I); cross hands are zero.
    """
    from ..services.interferometry.predict import predict_visibilities

    typer.echo(f"Predicting visibilities from model: {model}")
    typer.echo(f"Copying measurement set: {input_ms} -> {output_ms}")

    if output_ms.exists():
        fail(f"Output MS already exists: {output_ms}")

    planes, pixsize_x, pixsize_y = _model_planes(model)
    try:
        casacore_table = import_casacore_tables()
    except MissingDependencyError as exc:
        fail(str(exc))
    shutil.copytree(input_ms, output_ms)
    nthreads = int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1))

    with open_table(output_ms / "DATA_DESCRIPTION", opener=casacore_table) as data_description:
        spw_of_dd = spw_of_data_description(data_description)
        pol_of_dd = polarization_of_data_description(data_description)
    with open_table(output_ms / "SPECTRAL_WINDOW", opener=casacore_table) as spectral_window:
        chan_freq = [
            np.asarray(spectral_window.getcell("CHAN_FREQ", i), dtype=np.float64)
            for i in range(spectral_window.nrows())
        ]
    with open_table(output_ms / "POLARIZATION", opener=casacore_table) as polarization:
        corr_types = [
            np.asarray(polarization.getcell("CORR_TYPE", i)) for i in range(polarization.nrows())
        ]

    main_table = casacore_table(str(output_ms), readonly=False, ack=False)
    try:
        if "MODEL_DATA" not in main_table.colnames():
            from casacore.tables import makearrcoldesc, maketabdesc

            main_table.addcols(
                maketabdesc(makearrcoldesc("MODEL_DATA", 0j, ndim=2, valuetype="complex"))
            )
        ddids = main_table.getcol("DATA_DESC_ID")
        for dd in np.unique(ddids):
            freq = chan_freq[int(spw_of_dd[dd])]
            hands = np.isin(corr_types[int(pol_of_dd[dd])], list(_PARALLEL_HANDS))
            if len(planes) not in (1, len(freq)):
                raise ValueError(
                    f"model {model} has {len(planes)} planes; spectral window "
                    f"{int(spw_of_dd[dd])} has {len(freq)} channels (need 1 or {len(freq)})"
                )
            rows = np.flatnonzero(ddids == dd)
            selection = main_table.selectrows(rows.tolist())
            uvw = selection.getcol("UVW").astype(np.float64)
            vis = predict_visibilities(uvw, freq, planes, pixsize_x, pixsize_y, nthreads=nthreads)
            model_data = np.zeros(vis.shape + (len(hands),), dtype=np.complex64)
            model_data[:, :, hands] = vis[:, :, None]
            selection.putcol("MODEL_DATA", model_data)  # (nrows, nchan, ncorr)
            typer.echo(f"DATA_DESC_ID {int(dd)}: predicted {len(rows)} rows x {len(freq)} channels")
    finally:
        main_table.close()

    typer.echo(f"Wrote MODEL_DATA to: {output_ms}")


@image_app.command("predict-batch", hidden=True)
def predict_batch(
    imaging_parameters: Path = typer.Argument(..., help="Imaging parameter file"),
    output_directory: Path = typer.Argument(..., help="Output directory path"),
    num_cores: int = typer.Option(
        help="Number of cores per predict task", default=1, min=1, rich_help_panel="Cluster"
    ),
    use_slurm: bool = typer.Option(
        help="Whether or not to use slurm or not", default=True, rich_help_panel="Cluster"
    ),
    max_cores_per_node: MaxCoresPerNode = 95,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "02:00:00",
    slurm_memory: SlurmMemory = "16GB",
    slurm_n_jobs: int = typer.Option(
        1,
        "--slurm-workers",
        "--slurm-n-jobs",
        min=1,
        help="Number of Slurm workers (jobs).",
        rich_help_panel="Cluster",
    ),
    scheduler_host: SchedulerHost = None,
    scheduler_interface: SchedulerInterface = None,
    task_timeout: TaskTimeout = 3600,
):
    parameters = pd.read_csv(str(imaging_parameters))
    slurm_commands: list[tuple[str, list[str]]] = []
    for _, dset_parameter in tqdm(parameters.iterrows(), total=len(parameters)):
        input_filename = Path(dset_parameter["filename"])
        spw_id = int(dset_parameter["spectral_window_id"])
        # The directory image-from-ms wrote this row's products to (SPW-<n>_FIELD-<f>).
        model_path = task_directory(output_directory, dset_parameter)[0] / "wsclean-model.fits"
        if not model_path.exists():
            typer.echo(
                f"Missing model FITS, skipping row: ms={input_filename} spw={spw_id} "
                f"path={model_path}",
                err=True,
            )
            continue

        output_ms = model_path.parent / f"{input_filename.name}.predicted"
        predict_cmd = [
            "almasim",
            "image",
            "predict-single",
            str(input_filename.absolute()),
            str(model_path),
            str(output_ms),
        ]
        if use_slurm:
            label = input_filename.stem + f"_{spw_id}_predict"
            slurm_commands.append((label, predict_cmd))
        else:
            # Set up environment to prioritize system libraries for compatibility.
            cmd_env = os.environ.copy()
            ld_library_path = "/lib64:/usr/lib64:/usr/local/lib64:/lib:/usr/lib:/usr/local/lib"
            existing_ld = cmd_env.get("LD_LIBRARY_PATH", "")
            if existing_ld:
                ld_library_path = f"{ld_library_path}:{existing_ld}"
            cmd_env["LD_LIBRARY_PATH"] = ld_library_path
            subprocess.run(predict_cmd, check=True, env=cmd_env)

    if use_slurm:
        _run_commands_with_slurm_cluster(
            slurm_commands,
            cores_per_task=num_cores,
            node_cores=max_cores_per_node,
            queue=slurm_queue,
            project=slurm_project,
            walltime=slurm_walltime,
            memory=slurm_memory,
            n_jobs=slurm_n_jobs,
            scheduler_host=scheduler_host,
            scheduler_interface=scheduler_interface,
            task_timeout=task_timeout,
        )
