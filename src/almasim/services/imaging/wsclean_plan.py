"""Imaging tasks and WSClean command lines from the geometry rows (MADIV-28 T3.6).

Moved out of ``cli/image.py``: the cube channel choice, the per-row command
arguments, the task directories and the cube plan.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from almasim.services.imaging.ms_geometry import SPECTRAL_COLUMNS, image_pixels
from almasim.services.imaging.provenance import Provenance, find_ous_index
from almasim.services.instrument.constants import SPEED_OF_LIGHT_M_S


@dataclass(frozen=True)
class CubeOptions:
    """How ``cube_channels`` picks the number of output channels.

    * ``spectral_sampling``: output channels per spectral resolution element,
      the spectral counterpart of ``beam_sampling``. 2 (Nyquist) keeps the
      native channels of a Hanning-smoothed window.
    * ``velocity_resolution_km_s``: never make channels narrower than this
      (0: no limit).
    * ``max_cube_gb``: largest cube, in GB of 32-bit pixels, per product
      (image, residual ...); wide images get fewer channels.
    """

    spectral_sampling: float = 2.0
    velocity_resolution_km_s: float = 0.0
    max_cube_gb: float = 4.0


def _largest_divisor_at_most(n: int, limit: int) -> int:
    """The largest divisor of ``n`` not above ``limit`` (at least 1)."""
    for candidate in range(max(min(int(limit), int(n)), 1), 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def cube_channels(row: pd.Series, n_pixels: int, options: CubeOptions) -> int:
    """Output channels (WSClean ``-channels-out``) for one spectral window.

    The channel width is the widest of: the native channel, the spectral
    resolution divided by ``spectral_sampling``, and ``velocity_resolution``
    at the window's reference frequency. The count it gives is then capped so
    that an ``n_pixels`` x ``n_pixels`` cube stays below ``max_cube_gb``, and
    rounded down to a divisor of the native channel count, so every output
    channel averages the same number of native channels and the cube has one
    channel width.
    """
    missing = [column for column in SPECTRAL_COLUMNS if column not in row.index]
    if missing:
        raise ValueError(
            f"Cube imaging needs the spectral columns {', '.join(missing)}; "
            "rerun compute-parameters or use add_spectral_columns"
        )
    n_native = int(row["n_channels"])
    width = float(row["channel_width_hz"])
    resolution = float(row["spectral_resolution_hz"]) or width
    target_width = max(width, resolution / options.spectral_sampling)
    if options.velocity_resolution_km_s > 0:
        velocity_width = (options.velocity_resolution_km_s * 1e3 / SPEED_OF_LIGHT_M_S) * float(
            row["reference_frequency"]
        )
        target_width = max(target_width, velocity_width)
    wanted = max(int(math.floor(n_native * width / target_width + 1e-9)), 1)
    plane_bytes = 4 * n_pixels * n_pixels
    affordable = (
        wanted
        if math.isinf(options.max_cube_gb)
        else max(int(options.max_cube_gb * 1e9 // plane_bytes), 1)
    )
    return _largest_divisor_at_most(n_native, min(wanted, affordable))


def imaging_parameter_to_command_arg(
    imaging_parameters: pd.Series,
    fov_fraction: float,
    beam_sampling: float,
    auto_threshold: float | None = None,
    auto_mask: float | None = None,
    cube: CubeOptions | None = None,
    min_beams_across: float = 0.0,
    multiscale: bool = False,
    join_channels: bool = False,
) -> list[str]:
    """WSClean geometry and deconvolution flags for one spectral window.

    ``auto_threshold`` / ``auto_mask`` (in units of the residual noise) give
    CLEAN a stopping point; without them ``-niter`` is the only limit and every
    task cleans into the noise until the iteration cap.

    With ``cube`` the window is imaged as a spectral cube of ``cube_channels``
    channels (``-channels-out``, each channel cleaned on its own); without it,
    as one continuum plane. ``multiscale`` cleans with WSClean's multi-scale
    algorithm, which models extended emission that delta components leave
    in the residual. ``join_channels`` makes a cube find its components on
    the sum of the channels (WSClean ``-join-channels``) and fit them per
    channel, so emission too faint to clean in one channel is still cleaned.
    """
    spw = imaging_parameters["spectral_window_id"]
    synthetized_beam_size = imaging_parameters["synthetized_beam_size"] / beam_sampling
    n_pixels = image_pixels(imaging_parameters, fov_fraction, beam_sampling, min_beams_across)
    cmd_args = [
        "-scale",
        f"{synthetized_beam_size}asec",
        "-size",
        str(n_pixels),
        str(n_pixels),
        "-spws",
        str(spw),
        "-mgain",
        "0.85",
        "-niter",
        "100000",
        "-pol",
        "I",
        "-make-psf",
        "-weight",
        "briggs",
        "0.5",
    ]
    if auto_mask is not None and auto_mask > 0:
        cmd_args += ["-auto-mask", str(auto_mask)]
    if auto_threshold is not None and auto_threshold > 0:
        cmd_args += ["-auto-threshold", str(auto_threshold)]
    if multiscale:
        cmd_args.append("-multiscale")
    if cube is not None:
        channels = cube_channels(imaging_parameters, n_pixels, cube)
        if channels > 1:
            cmd_args += ["-channels-out", str(channels)]
            if join_channels:
                cmd_args.append("-join-channels")
    return cmd_args


@dataclass(frozen=True)
class ImagingTask:
    """One imaging task: one MeasurementSet, one spectral window (and field).

    ``command`` makes the continuum image (empty when only a cube is made);
    ``cube_command``, when set, the spectral cube.
    """

    label: str
    ms_path: Path
    spw: int
    output_dir: Path
    command: list[str]
    field_ids: tuple[int, ...] = ()
    cube_command: list[str] | None = None
    dish_diameter: float | None = None
    provenance: Provenance | None = None


def _field_arguments(row: pd.Series) -> list[str]:
    """``-field`` selection from the ``target_field_ids`` column, if the CSV has one."""
    value = row.get("target_field_ids") if hasattr(row, "get") else None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.endswith(".0"):  # a single id that pandas read as a float
        text = text[:-2]
    return ["-field", text]


def row_dish_diameter(row: pd.Series) -> float | None:
    """The dish diameter (m) behind a parameter row's ``fov_per_frequency``.

    ``compute_imaging_parameters`` sets it to the primary beam's half-power
    width for the smallest dish, so D comes back exactly (12 or 7 m for ALMA). None when
    the row lacks the field of view or the frequency.
    """
    from almasim.services.imaging.primary_beam import dish_diameter_from_fov

    try:
        diameter = dish_diameter_from_fov(
            float(row["fov_per_frequency"]), float(row["reference_frequency"])
        )
    except (KeyError, TypeError, ValueError):
        return None
    return round(diameter, 2) if math.isfinite(diameter) else None


def row_provenance(
    row: pd.Series,
    ous_index_csv: str | Path | None = None,
    indexes: dict[Path | None, dict[str, dict[str, str]]] | None = None,
) -> Provenance:
    """A parameter row's provenance: its MS, window and field, and the OUSs of its EB.

    The OUSs come from ``ous_index_csv``, or from the ``ous_index.csv`` nearest
    above the MS when it is None (none: the OUSs are left out). ``indexes``
    keeps the indexes read, by path, across the rows of one run.
    """
    from almasim.services.archive.delivery import read_ous_index

    ms = Path(row["filename"]).expanduser().absolute()
    path = Path(ous_index_csv) if ous_index_csv else find_ous_index(ms)
    indexes = {} if indexes is None else indexes
    if path not in indexes:
        indexes[path] = read_ous_index(path) if path is not None else {}
    field = row.get("field_name") if hasattr(row, "get") else None
    if field is None or (isinstance(field, float) and math.isnan(field)):
        field = ""
    return Provenance.for_ms(
        [ms],
        indexes[path],
        spw=int(row["spectral_window_id"]),
        field=str(field),
    )


def task_directory(output_directory: Path, row: pd.Series) -> tuple[Path, tuple[int, ...]]:
    """A parameter row's task directory under ``output_directory`` and its field ids.

    One field per row gets its own directory and markers, so the fields of
    one window never share an image; rows without a field, or with a list
    (CSVs from before per-field rows), keep SPW-<n>.
    """
    input_filename = Path(row["filename"]).expanduser().absolute()
    spw = int(row["spectral_window_id"])
    field_args = _field_arguments(row)
    field_ids = tuple(int(f) for f in field_args[1].split(",")) if field_args else ()
    single_field = field_ids[0] if len(field_ids) == 1 else None
    task_name = f"SPW-{spw}" if single_field is None else f"SPW-{spw}_FIELD-{single_field}"
    return Path(output_directory) / input_filename.stem / task_name, field_ids


def build_imaging_tasks(
    parameters: pd.DataFrame,
    output_directory: Path,
    *,
    fov_fraction: float,
    beam_sampling: float,
    num_cores: int,
    max_cores_per_node: int,
    wsclean_bin: str = "wsclean",
    overwrite_outputs: bool = False,
    trust_existing_images: bool = False,
    auto_threshold: float | None = None,
    auto_mask: float | None = None,
    update_model: bool = False,
    task_memory_gb: float | None = None,
    scratch_dir: str | None = None,
    single_window: bool = True,
    continuum: bool = True,
    cube: CubeOptions | None = None,
    cube_products: Sequence[str] | None = None,
    cube_reorder: bool = True,
    min_beams_across: float = 0.0,
    multiscale: bool = False,
    join_channels: bool = False,
    ous_index_csv: str | Path | None = None,
) -> tuple[list[ImagingTask], int]:
    """Turn parameter rows into WSClean tasks, skipping the ones already done.

    A task makes the continuum image (``continuum``, ``wsclean-*.fits``)
    and/or, with ``cube``, a spectral cube of ``cube_channels`` channels
    (``wsclean-cube-*.fits``); both runs share the task's single-window MS.
    The cube gets a dirty image only when ``cube_products`` asks for it.

    The cube run lets WSClean reorder (``cube_reorder``) even on a
    single-window MS: with ``-no-reorder`` it reads the whole MS about four
    times per output channel, and a 480-channel Band 1 cube
    (uid___A002_X1142854_X18e99 SPW 25) took about 2 h against 4.5 min
    reordered, with the same images (2026-10-13). The 3.7 reordering crash
    needs several windows in the MS; the continuum run keeps ``-no-reorder``.

    Returns ``(tasks, skipped)``. A task is done when its ``SPW-<n>.done``
    marker and every image it makes exist, so a cube run over a folder of
    continuum images (``continuum=False``) redoes only the missing cubes;
    ``trust_existing_images`` also accepts a continuum image produced before
    markers existed and writes its marker. ``overwrite_outputs`` redoes
    everything.

    Each task carries its data's provenance (``row_provenance``) for the
    products' FITS headers: the OUSs from ``ous_index_csv``, or from the
    ``ous_index.csv`` nearest above each MS when it is None.
    """
    from almasim.services.imaging.markers import is_imaging_complete, write_imaging_marker
    from almasim.services.imaging.products import (
        CUBE_IMAGE_FILENAME,
        CUBE_NAME,
        IMAGE_FILENAME,
        SCRATCH_PLACEHOLDER,
        SINGLE_WINDOW_PLACEHOLDER,
    )
    from almasim.services.imaging.wsclean.command import WSCleanCommand

    if not continuum and cube is None:
        raise ValueError("Nothing to image: neither the continuum nor a cube was requested")
    required = (IMAGE_FILENAME,) * continuum + (CUBE_IMAGE_FILENAME,) * (cube is not None)
    tasks: list[ImagingTask] = []
    skipped = 0
    # WSClean's -mem is a PERCENTAGE of system memory. Passing the fraction
    # (0.1 for 10 cores of 96) limited every task to 0.1 % ≈ 0.4 GB and made
    # WSClean segfault on the first gridding pass.
    mem_percent = min(100.0, 100.0 * num_cores / max_cores_per_node)
    # WSClean runs with its task directory as cwd, so a relative path in the
    # command resolves against that directory, not the caller's: a run started
    # with output "ms-imaged-flagged" looked for its single-window MS under
    # .../SPW-5/ms-imaged-flagged/... and every task failed (2026-10-09).
    output_directory = Path(output_directory).expanduser().absolute()
    indexes: dict[Path | None, dict[str, dict[str, str]]] = {}
    for _, row in parameters.iterrows():
        input_filename = Path(row["filename"]).expanduser().absolute()
        spw = int(row["spectral_window_id"])
        outdir, field_ids = task_directory(output_directory, row)
        single_field = field_ids[0] if len(field_ids) == 1 else None
        if not overwrite_outputs:
            if is_imaging_complete(outdir, required):
                skipped += 1
                continue
            if (
                trust_existing_images
                and required == (IMAGE_FILENAME,)
                and (outdir / IMAGE_FILENAME).is_file()
            ):
                write_imaging_marker(outdir, ms_path=str(input_filename), spw=spw)
                skipped += 1
                continue

        # The loop variables are bound as defaults: ruff cannot see that the
        # closure runs within this iteration only.
        def wsclean_command(
            name: str,
            with_cube: bool,
            row: pd.Series = row,
            outdir: Path = outdir,
            input_filename: Path = input_filename,
        ) -> list[str]:
            command_args = imaging_parameter_to_command_arg(
                row,
                fov_fraction,
                beam_sampling,
                auto_threshold=auto_threshold,
                auto_mask=auto_mask,
                cube=cube if with_cube else None,
                min_beams_across=min_beams_across,
                multiscale=multiscale,
                join_channels=join_channels,
            )
            if with_cube and cube_products is not None and "dirty" not in cube_products:
                command_args.append("-no-dirty")
            command = WSCleanCommand(
                wsclean_bin,
                (
                    "-name",
                    str(outdir / name),
                    "-j",
                    str(num_cores),
                    *command_args,
                    # An absolute limit when the caller knows the worker's
                    # allocation (WSClean's -mem is a share of the *node*, which
                    # can be far more than the Slurm job may use); otherwise the
                    # node share.
                    *(
                        ["-abs-mem", f"{task_memory_gb:g}"]
                        if task_memory_gb
                        else ["-mem", f"{mem_percent:g}"]
                    ),
                    # Reordered visibilities go next to the task output (or into a
                    # per-task scratch directory), not next to the MS: several
                    # spectral windows of one MS run at the same time and would
                    # otherwise clobber each other's <ms>-part*.tmp files.
                    "-temp-dir",
                    SCRATCH_PLACEHOLDER if scratch_dir else str(outdir),
                    # Writing the model back into MODEL_DATA makes concurrent tasks
                    # write the same MS from different nodes and grows every
                    # calibrated MS; the model image is on disk anyway.
                    "-update-model-required" if update_model else "-no-update-model-required",
                    # -field is needed on the single-window MS too: WSClean images
                    # field 0 by default and the extracted MS holds the target
                    # fields under their original ids.
                    *_field_arguments(row),
                ),
                str(input_filename),
            )
            if single_window:
                # The wrapper hands WSClean a per-task MS holding only this
                # window and the target fields, so no -spws selection and no
                # reordering (WSClean 3.7's reordering reads out of bounds on
                # multi-window ALMA sets; see extract_single_window_ms).
                command = replace(command.without("-spws"), ms=SINGLE_WINDOW_PLACEHOLDER)
                if not (with_cube and cube_reorder):
                    command = command.with_option("-no-reorder")
            return command.argv()

        tasks.append(
            ImagingTask(
                label=f"{input_filename.stem}_{spw}"
                + ("" if single_field is None else f"_f{single_field}"),
                ms_path=input_filename,
                spw=spw,
                output_dir=outdir,
                command=wsclean_command("wsclean", False) if continuum else [],
                field_ids=field_ids,
                cube_command=wsclean_command(CUBE_NAME, True) if cube is not None else None,
                dish_diameter=row_dish_diameter(row),
                provenance=row_provenance(row, ous_index_csv, indexes),
            )
        )
    return tasks, skipped


CUBE_PRODUCTS = ("image", "residual", "psf", "model", "dirty")


def cube_plan(
    parameters: pd.DataFrame,
    fov_fraction: float,
    beam_sampling: float,
    options: CubeOptions,
    min_beams_across: float = 0.0,
) -> pd.DataFrame:
    """Per row: image side, native and cube channels, and one cube's size in GB.

    ``size_limited`` marks the rows that ``max_cube_gb`` gave fewer channels
    than the spectral sampling asked for.
    """
    unlimited = replace(options, max_cube_gb=math.inf)
    pixels = parameters.apply(
        lambda row: image_pixels(row, fov_fraction, beam_sampling, min_beams_across), axis=1
    )
    channels, wanted = [], []
    for (_, row), n in zip(parameters.iterrows(), pixels, strict=True):
        channels.append(cube_channels(row, int(n), options))
        wanted.append(cube_channels(row, int(n), unlimited))
    plan = pd.DataFrame(
        {
            "n_pixels": pixels.astype(int).to_numpy(),
            "n_channels": parameters["n_channels"].astype(int).to_numpy(),
            "cube_channels": channels,
            "size_limited": np.array(channels) < np.array(wanted),
        },
        index=parameters.index,
    )
    plan["cube_gb"] = 4 * plan["n_pixels"] ** 2 * plan["cube_channels"] / 1e9
    return plan
