"""Noise models: what is released, what each component does, the file schema (MADIV-28).

``noise show`` lists the released models, or prints one (a released name, a
candidate file) with the components it applies; ``noise components`` lists
the components a model can hold and their parameters; ``noise schema``
writes the JSON Schema model files validate against. ``noise windows`` makes
the table of an imaging run's windows; ``noise extract`` measures real
windows' noise, one window or a table of them, on Slurm.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from ..services.noise_model.components import NoiseComponent
from ._options import (
    Backend,
    PostprocessBackend,
    SchedulerHost,
    SchedulerInterface,
    SlurmCores,
    SlurmMemory,
    SlurmOptions,
    SlurmProject,
    SlurmQueue,
    SlurmWalltime,
    TaskTimeout,
)
from ._shared import fail, usage_error

# A chain of steps (the console's Noise models page) runs each after the one
# before succeeded; one window failing would stop it for all the others.
AllowPartial = typer.Option(
    False,
    "--allow-partial",
    help="Succeed when some windows fail and at least one did not (failures stay listed).",
    rich_help_panel="Outputs",
)


def _failed(failures: dict[str, str], tasks: list, allow_partial: bool) -> bool:
    """Whether the step failed: any window, or all of them with ``allow_partial``."""
    return bool(failures) and (not allow_partial or len(failures) >= len(tasks))


noise_app = typer.Typer(
    help="Noise models: released versions, components, schema; measuring real data's noise.",
    no_args_is_help=True,
)


@noise_app.command("show")
def show(
    model: str | None = typer.Argument(
        None,
        help=(
            "A released model ('alma', 'alma@1'), or a JSON file; omitted: list the "
            "released models."
        ),
    ),
    only: str | None = typer.Option(
        None, "--noise-only", help="Show the model with only these components (comma-separated)."
    ),
    without: str | None = typer.Option(
        None, "--noise-without", help="Show the model without these components (comma-separated)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the model's JSON."),
) -> None:
    """List the released noise models, or show one."""
    from ..services.noise_model import released, resolve_noise_model

    if model is None:
        for name, versions in sorted(released().items()):
            newest = versions[max(versions)]
            typer.echo(
                f"{name}: versions {', '.join(map(str, sorted(versions)))}; "
                f"newest applies {', '.join(newest.enabled()) or 'nothing'}"
            )
        return
    try:
        resolved = resolve_noise_model(model)
        if resolved is None:
            typer.echo("none: applies nothing")
            return
        resolved = resolved.select(
            only=[only] if only is not None else None, without=[without] if without else None
        )
    except (ValueError, TypeError) as exc:
        fail(f"noise model {model!r}: {exc}")
    if as_json:
        typer.echo(resolved.model_dump_json(indent=2))
        return
    typer.echo(f"{resolved.label()}  sha256 {resolved.sha256()}")
    if resolved.description:
        typer.echo(resolved.description)
    if resolved.provenance is not None:
        p = resolved.provenance
        typer.echo(f"provenance: {p.method}" + (f"; {p.data}" if p.data else ""))
    typer.echo(f"settings: {_values(resolved.settings.model_dump())}")
    for name, component in resolved.components.items():
        params = component.model_dump(exclude={"enabled"})
        state = "on " if component.enabled else "off"
        typer.echo(f"  [{state}] {name}" + (f": {_values(params)}" if params else ""))


@noise_app.command("components")
def components() -> None:
    """The components a noise model can hold, and their parameters."""
    from ..services.noise_model import Components

    for name, info in Components.model_fields.items():
        cls = _component_class(info.annotation)
        typer.echo(f"{name}: {' '.join((cls.__doc__ or '').split())}")
        for field, spec in cls.model_fields.items():
            if field == "enabled":
                continue
            default = "required" if spec.is_required() else f"default {spec.default}"
            typer.echo(f"    {field} ({default}): {spec.description or ''}")


@noise_app.command("schema")
def schema(
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write it here (default: print)."
    ),
) -> None:
    """The JSON Schema of a noise model file."""
    from ..services.noise_model.schema import noise_model_schema

    text = json.dumps(noise_model_schema(), indent=2) + "\n"
    if output is None:
        typer.echo(text, nl=False)
    else:
        output.write_text(text)
        typer.echo(f"wrote {output}")


def _component_class(annotation: object) -> type[NoiseComponent]:
    """The component class of a ``Components`` field (``X | None``)."""
    for arg in getattr(annotation, "__args__", ()):
        if isinstance(arg, type) and issubclass(arg, NoiseComponent):
            return arg
    raise TypeError(f"not a component field: {annotation!r}")


def _values(values: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in values.items())


@noise_app.command("windows")
def windows(
    parameters: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="The imaging run's parameters CSV (image compute-parameters').",
    ),
    images: Path = typer.Option(
        ...,
        "--images",
        exists=True,
        file_okay=False,
        help="The imaging run's output folder (image image-from-ms').",
        rich_help_panel="Inputs",
    ),
    metadata: Path = typer.Option(
        ...,
        "--metadata",
        exists=True,
        dir_okay=False,
        help="The archive search the data came from (an almasim metadata CSV).",
        rich_help_panel="Inputs",
    ),
    products: list[Path] | None = typer.Option(
        None,
        "--products",
        exists=True,
        dir_okay=False,
        help=(
            "Resolved-products CSV(s) of those observations, which link each MS's "
            "execution block to its observation (default: *products*.csv beside --metadata)."
        ),
        rich_help_panel="Inputs",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Where windows.csv and rows.csv go.",
        rich_help_panel="Outputs",
    ),
    done_only: bool = typer.Option(
        True,
        "--done-only/--all-tasks",
        help="Only windows whose imaging task finished (its .done marker).",
        rich_help_panel="Inputs",
    ),
    science_only: bool = typer.Option(
        True,
        "--science-only/--all-fields",
        help="Only science target fields (OBSERVE_TARGET), read from each MS.",
        rich_help_panel="Inputs",
    ),
    max_windows: int | None = typer.Option(
        None, "--max-windows", min=1, help="Keep the first N windows.", rich_help_panel="Inputs"
    ),
    names: str | None = typer.Option(
        None,
        "--names",
        help="Only these windows (comma-separated names, as a previous run listed them).",
        rich_help_panel="Inputs",
    ),
) -> None:
    """Write the windows table (windows.csv) and archive rows (rows.csv) of an
    imaging run, for noise extract, fit and validate."""
    import pandas as pd

    from ..services.imaging.ms_geometry import science_selection
    from ..services.noise.windows import imaging_windows
    from ..services.simulation.input_cubes import (
        observations_by_execution_block,
        products_csvs_near,
    )

    product_csvs = list(products or products_csvs_near(metadata))
    observations = observations_by_execution_block(product_csvs)
    if not observations:
        usage_error(
            "--products: no resolved-products CSV links the MSs to their observations "
            "(give the products CSV of the download)"
        )
    found = imaging_windows(
        pd.read_csv(parameters),
        images.expanduser().resolve(),
        pd.read_csv(metadata, dtype=str, keep_default_na=False),
        observations,
        done_only=done_only,
        science_fields=(lambda ms: science_selection(Path(ms))[1]) if science_only else None,
        max_windows=max_windows,
        names={n.strip() for n in names.split(",") if n.strip()} if names else None,
    )
    for name, why in found.skipped:
        typer.echo(f"  left out {name}: {why}")
    if found.windows.empty:
        fail("No window to measure; see the reasons above.")
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    found.windows.to_csv(out / "windows.csv", index=False)
    found.rows.to_csv(out / "rows.csv", index=False)
    typer.echo(
        f"{len(found.windows)} window(s) -> {out / 'windows.csv'}, their archive rows -> "
        f"{out / 'rows.csv'}"
    )


@noise_app.command("extract")
def extract(
    ms: Path | None = typer.Argument(
        None, help="A calibrated MeasurementSet (*.ms.split.cal); or give --windows."
    ),
    out: Path = typer.Option(
        ..., "--out", help="Where the measurements go (windows/<name>/).", rich_help_panel="Outputs"
    ),
    spw: int | None = typer.Option(
        None, "--spw", min=0, help="The MS's spectral window.", rich_help_panel="Inputs"
    ),
    field: str | None = typer.Option(
        None, "--field", help="Field name or id (default: every field).", rich_help_panel="Inputs"
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help="The window's name (default: uid, spw, field).",
        rich_help_panel="Outputs",
    ),
    syscal_ms: Path | None = typer.Option(
        None,
        "--syscal-ms",
        help="Where to read SYSCAL's Tsys (default: the MS).",
        rich_help_panel="Inputs",
    ),
    line_channels: str | None = typer.Option(
        None,
        "--line-channels",
        help="Channels of line emission to leave out ('a~b;c').",
        rich_help_panel="Inputs",
    ),
    task_log: Path | None = typer.Option(
        None,
        "--task-log",
        help="An ALMASim imaging task's log; its 'lines in channels' are left out.",
        rich_help_panel="Inputs",
    ),
    pwv: float | None = typer.Option(
        None,
        "--pwv",
        min=0,
        help="The observation's precipitable water vapour (mm; the archive's PWV), "
        "for noise aggregate's receiver temperatures.",
        rich_help_panel="Inputs",
    ),
    windows: Path | None = typer.Option(
        None,
        "--windows",
        help="A CSV of windows (columns ms, spw; optional field, name, pwv, syscal_ms, "
        "line_channels, task_log), measured one per task.",
        rich_help_panel="Inputs",
    ),
    max_visibilities: int = typer.Option(
        40_000_000,
        "--max-visibilities",
        min=1000,
        help="Rows are sampled evenly over the observation up to about this many visibilities.",
        rich_help_panel="Inputs",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Measure windows again even when measured.",
        rich_help_panel="Outputs",
    ),
    allow_partial: bool = AllowPartial,
    backend: PostprocessBackend = Backend.sync,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "02:00:00",
    slurm_cores: SlurmCores = 2,
    slurm_memory: SlurmMemory = "32GB",
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=1,
        help="Slurm workers measuring windows at once.",
        rich_help_panel="Cluster",
    ),
    scheduler_host: SchedulerHost = None,
    scheduler_interface: SchedulerInterface = None,
    task_timeout: TaskTimeout = 4 * 3600.0,
) -> None:
    """Measure real windows' noise: the level against WEIGHT, the channel
    correlation, ALMA's Tsys across the window, one visibility's noise and
    each antenna's noise shape (window.json)."""
    from functools import partial

    from ..services.noise.batch import run_in_process, run_on_slurm, run_tasks

    tasks = _extract_tasks(ms, windows, spw, field, name, syscal_ms, line_channels, task_log, pwv)
    out = out.expanduser().resolve()
    if backend == Backend.slurm and len(tasks) > 1:
        options = SlurmOptions(
            queue=slurm_queue,
            project=slurm_project,
            walltime=slurm_walltime,
            memory=slurm_memory,
            n_workers=slurm_workers,
            scheduler_host=scheduler_host,
            scheduler_interface=scheduler_interface,
        )
        runner = partial(
            run_on_slurm,
            out=out,
            backend_kwargs={"node_cores": slurm_cores + 1, **options.backend_kwargs()},
            cores=slurm_cores,
            timeout=task_timeout,
            extra_args=["--max-visibilities", str(max_visibilities)]
            + (["--overwrite"] if overwrite else []),
            echo=typer.echo,
        )
    else:
        runner = partial(
            run_in_process, out=out, max_visibilities=max_visibilities, echo=typer.echo
        )
    failures = run_tasks(tasks, out, runner, overwrite=overwrite, echo=typer.echo)
    measured = len(tasks) - len(failures)
    typer.echo(
        f"{measured} of {len(tasks)} window(s) measured; failures in {out / 'failures.json'}"
    )
    if _failed(failures, tasks, allow_partial):
        raise typer.Exit(code=1)


def _extract_tasks(
    ms: Path | None,
    windows: Path | None,
    spw: int | None,
    field: str | None,
    name: str | None,
    syscal_ms: Path | None,
    line_channels: str | None,
    task_log: Path | None,
    pwv: float | None = None,
) -> list:
    """The windows ``noise extract`` measures: one MS's, or a table's."""
    from ..services.noise.batch import WindowTask, read_windows, window_name

    if (ms is None) == (windows is None):
        usage_error("give an MS (with --spw) or --windows, not both")
    if windows is not None:
        if any(v is not None for v in (spw, field, name, syscal_ms, line_channels, task_log, pwv)):
            usage_error(
                "--spw, --field, --name, --syscal-ms, --line-channels and --task-log "
                "describe one MS; a --windows table has them as columns"
            )
        try:
            return read_windows(windows)
        except (OSError, ValueError) as exc:
            usage_error(f"--windows: {exc}")
    if ms is None or spw is None:
        usage_error("--spw is needed with an MS")
    return [
        WindowTask(
            ms=str(ms.expanduser().resolve()),
            spw=spw,
            name=name or window_name(str(ms), spw, field),
            field=field,
            syscal_ms=str(syscal_ms.expanduser().resolve()) if syscal_ms else None,
            line_channels=line_channels,
            task_log=str(task_log) if task_log else None,
            pwv=None if pwv is None else str(pwv),
        )
    ]


@noise_app.command("fit")
def fit(
    ms: Path | None = typer.Argument(
        None, help="A calibrated MeasurementSet (*.ms.split.cal); or give --windows."
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="The noise extract output (windows/<name>/window.json); fit.json goes beside it.",
        rich_help_panel="Outputs",
    ),
    spw: int | None = typer.Option(
        None, "--spw", min=0, help="The MS's spectral window.", rich_help_panel="Inputs"
    ),
    field: str | None = typer.Option(
        None, "--field", help="Field name or id (default: every field).", rich_help_panel="Inputs"
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help="The window's name, as noise extract gave it.",
        rich_help_panel="Inputs",
    ),
    task_dir: Path | None = typer.Option(
        None,
        "--task-dir",
        help="The window's ALMASim imaging task (its clean models and quality.json).",
        rich_help_panel="Inputs",
    ),
    windows: Path | None = typer.Option(
        None,
        "--windows",
        help="A CSV of windows, as for noise extract, with a task_dir column.",
        rich_help_panel="Inputs",
    ),
    continuum: str = typer.Option(
        "masked",
        "--continuum-model",
        help="The continuum clean model: 'masked' (where the image is above 5 sigma) or 'all'.",
        rich_help_panel="Model",
    ),
    line: str = typer.Option(
        "masked",
        "--line-model",
        help=(
            "The cube clean model: 'masked' (line channels, inside the line mask), 'all' or 'none'."
        ),
        rich_help_panel="Model",
    ),
    realizations: int = typer.Option(
        3,
        "--realizations",
        min=1,
        help="Draws of the noise-only references.",
        rich_help_panel="Model",
    ),
    seed: int = typer.Option(
        0, "--seed", help="Seed of the references' draws.", rich_help_panel="Model"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Fit windows again even when fitted.", rich_help_panel="Outputs"
    ),
    allow_partial: bool = AllowPartial,
    backend: PostprocessBackend = Backend.sync,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "04:00:00",
    slurm_cores: SlurmCores = 4,
    slurm_memory: SlurmMemory = "64GB",
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=1,
        help="Slurm workers fitting windows at once.",
        rich_help_panel="Cluster",
    ),
    scheduler_host: SchedulerHost = None,
    scheduler_interface: SchedulerInterface = None,
    task_timeout: TaskTimeout = 8 * 3600.0,
) -> None:
    """Fit real windows' calibration residuals against their clean models: the
    slow and fast atmospheric phase, the antenna gains' and the baselines' own
    errors (fit.json), the values of the noise model's slow_phase, fast_phase,
    antenna_gains and baseline_errors components."""
    from functools import partial

    from ..services.noise.batch import (
        WindowTask,
        read_windows,
        run_in_process,
        run_on_slurm,
        run_tasks,
        window_name,
    )

    if continuum not in ("masked", "all") or line not in ("masked", "all", "none"):
        usage_error("--continuum-model takes masked or all; --line-model masked, all or none")
    if (ms is None) == (windows is None):
        usage_error("give an MS (with --spw and --task-dir) or --windows, not both")
    if windows is not None:
        if any(v is not None for v in (spw, field, name, task_dir)):
            usage_error(
                "--spw, --field, --name and --task-dir describe one MS; "
                "a --windows table has them as columns"
            )
        try:
            tasks = read_windows(windows)
        except (OSError, ValueError) as exc:
            usage_error(f"--windows: {exc}")
    else:
        if ms is None or spw is None or task_dir is None:
            usage_error("--spw and --task-dir are needed with an MS")
        tasks = [
            WindowTask(
                ms=str(ms.expanduser().resolve()),
                spw=spw,
                name=name or window_name(str(ms), spw, field),
                field=field,
                task_dir=str(task_dir.expanduser().resolve()),
            )
        ]
    out = out.expanduser().resolve()
    options: dict[str, Any] = {
        "continuum": continuum,
        "line": line,
        "realizations": realizations,
        "seed": seed,
    }
    if backend == Backend.slurm and len(tasks) > 1:
        slurm = SlurmOptions(
            queue=slurm_queue,
            project=slurm_project,
            walltime=slurm_walltime,
            memory=slurm_memory,
            n_workers=slurm_workers,
            scheduler_host=scheduler_host,
            scheduler_interface=scheduler_interface,
        )
        runner = partial(
            run_on_slurm,
            out=out,
            step="fit",
            backend_kwargs={"node_cores": slurm_cores + 1, **slurm.backend_kwargs()},
            cores=slurm_cores,
            timeout=task_timeout,
            extra_args=[
                "--continuum-model",
                continuum,
                "--line-model",
                line,
                "--realizations",
                str(realizations),
                "--seed",
                str(seed),
                *(["--overwrite"] if overwrite else []),
            ],
            echo=typer.echo,
        )
    else:
        runner = partial(run_in_process, out=out, step="fit", echo=typer.echo, **options)
    failures = run_tasks(tasks, out, runner, step="fit", overwrite=overwrite, echo=typer.echo)
    typer.echo(
        f"{len(tasks) - len(failures)} of {len(tasks)} window(s) fitted; "
        f"failures in {out / 'failures-fit.json'}"
    )
    if _failed(failures, tasks, allow_partial):
        raise typer.Exit(code=1)


@noise_app.command("aggregate")
def aggregate(
    out: Path = typer.Option(
        ...,
        "--out",
        help="The noise extract and fit output (windows/<name>/window.json, fit.json).",
        rich_help_panel="Inputs",
    ),
    base: str = typer.Option(
        "alma",
        "--from",
        help="The model whose values the windows' replace ('alma', 'alma@1' or a file).",
        rich_help_panel="Inputs",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Where the candidate goes (default: <out>/candidate.json).",
        rich_help_panel="Outputs",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help="The candidate's name (default: <base>-candidate).",
        rich_help_panel="Outputs",
    ),
    version: int | None = typer.Option(
        None,
        "--version",
        min=1,
        help="The candidate's version (default: the base's + 1).",
        rich_help_panel="Outputs",
    ),
    min_windows: int = typer.Option(
        3,
        "--min-windows",
        min=1,
        help="Windows a value needs; fewer keep the base's.",
        rich_help_panel="Inputs",
    ),
) -> None:
    """Reduce the windows' measurements into a candidate noise model: each value
    they measure is their median, with its window count and spread recorded."""
    from ..services.noise.aggregate import Measured
    from ..services.noise.aggregate import aggregate as reduce_windows
    from ..services.noise.batch import FIT_FILE, WINDOW_FILE
    from ..services.noise.measurements import WindowFit, WindowNoise
    from ..services.noise_model import resolve_noise_model

    out = out.expanduser().resolve()
    try:
        model = resolve_noise_model(base)
    except (ValueError, TypeError) as exc:
        usage_error(f"--from: {exc}")
    if model is None:
        usage_error("--from: a model, not none")
    measured = []
    for path in sorted((out / "windows").glob(f"*/{WINDOW_FILE}")):
        fit_path = path.with_name(FIT_FILE)
        measured.append(
            Measured(
                window=WindowNoise.model_validate_json(path.read_text()),
                fit=WindowFit.model_validate_json(fit_path.read_text())
                if fit_path.is_file()
                else None,
            )
        )
    if not measured:
        usage_error(f"--out: no windows/*/{WINDOW_FILE} under {out}; run noise extract first")
    candidate = reduce_windows(
        measured, model, name=name, version=version, data=str(out), min_windows=min_windows
    )
    target = (output or out / "candidate.json").expanduser()
    target.write_text(candidate.model_dump_json(indent=2) + "\n")
    fitted = sum(m.fit is not None for m in measured)
    typer.echo(f"{len(measured)} window(s), {fitted} fitted -> {candidate.label()} in {target}")
    provenance = candidate.provenance
    for key, stats in sorted(((provenance.values if provenance else None) or {}).items()):
        typer.echo(
            f"  {key}: {stats.median:.4g} ({stats.q16:.4g}-{stats.q84:.4g}, {stats.n} windows)"
        )
    if provenance is not None and provenance.notes:
        typer.echo(f"  kept: {provenance.notes}")


@noise_app.command("validate")
def validate(
    model: str = typer.Option(
        ...,
        "--model",
        help="The model to validate: a candidate file or a released name.",
        rich_help_panel="Inputs",
    ),
    windows: Path = typer.Option(
        ...,
        "--windows",
        help="The windows table noise extract measured (name, ms, spw; task_dir: their imaging).",
        rich_help_panel="Inputs",
    ),
    metadata: Path = typer.Option(
        ...,
        "--metadata",
        help="Each window's archive row (as 'almasim metadata' writes them) with a name column.",
        rich_help_panel="Inputs",
    ),
    measured: Path = typer.Option(
        ...,
        "--measured",
        help="noise extract's output for the real windows.",
        rich_help_panel="Inputs",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Where the validation goes (validation/<model sha>/).",
        rich_help_panel="Outputs",
    ),
    name: str | None = typer.Option(
        None, "--name", help="Validate only this window of the table.", rich_help_panel="Inputs"
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="The simulations' seed (per window, plus its name's).",
        rich_help_panel="Simulation",
    ),
    keep_ms: bool = typer.Option(
        False, "--keep-ms", help="Keep the simulated MSs.", rich_help_panel="Outputs"
    ),
    wsclean_bin: str | None = typer.Option(None, "--wsclean-bin", rich_help_panel="Imaging"),
    casa_data_root: str | None = typer.Option(None, "--casa-data-root", rich_help_panel="Imaging"),
    num_cores: int = typer.Option(8, "--num-cores", min=1, rich_help_panel="Imaging"),
    task_memory_gb: float = typer.Option(
        40.0, "--task-memory-gb", min=1.0, rich_help_panel="Imaging"
    ),
    imaging_timeout: float = typer.Option(
        10800.0, "--imaging-timeout", min=60, rich_help_panel="Imaging"
    ),
    criteria: Path | None = typer.Option(
        None,
        "--criteria",
        help="A JSON file of comparison criteria (noise compare's).",
        rich_help_panel="Inputs",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Validate windows again.", rich_help_panel="Outputs"
    ),
    allow_partial: bool = AllowPartial,
    judge: bool = typer.Option(
        True,
        "--judge/--no-judge",
        help="Compare the validated windows at the end.",
        rich_help_panel="Outputs",
    ),
    backend: PostprocessBackend = Backend.sync,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "08:00:00",
    slurm_memory: SlurmMemory = "64GB",
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=1,
        help="Slurm workers validating windows at once.",
        rich_help_panel="Cluster",
    ),
    scheduler_host: SchedulerHost = None,
    scheduler_interface: SchedulerInterface = None,
    task_timeout: TaskTimeout = 12 * 3600.0,
) -> None:
    """Simulate the real windows with a noise model, measure and image them as
    the real ones were, and compare (validation/<model sha>/comparison.json)."""
    from functools import partial

    from ..services.noise import validate as v
    from ..services.noise.batch import read_windows, run_tasks

    noise_model, tasks, rows, rules = _validate_inputs(model, windows, metadata, criteria, name)
    out, measured = out.expanduser().resolve(), measured.expanduser().resolve()
    imaging = v.Imaging(
        wsclean_bin=wsclean_bin,
        casa_data_root=casa_data_root,
        num_cores=num_cores,
        task_memory_gb=task_memory_gb,
        task_timeout=imaging_timeout,
    )
    run_here = partial(
        v.run_in_process,
        rows=rows,
        model=noise_model,
        measured=measured,
        out=out,
        imaging=imaging,
        seed=seed,
        keep_ms=keep_ms,
        echo=typer.echo,
    )

    def window_command(task: Any) -> list[str]:
        args = [
            "validate",
            "--model",
            model,
            "--windows",
            str(windows),
            "--metadata",
            str(metadata),
            "--measured",
            str(measured),
            "--out",
            str(out),
            "--name",
            task.name,
            "--seed",
            str(seed),
            "--num-cores",
            str(num_cores),
            "--task-memory-gb",
            str(task_memory_gb),
            "--imaging-timeout",
            str(imaging_timeout),
            "--no-judge",
        ]
        for flag, value in (("--wsclean-bin", wsclean_bin), ("--casa-data-root", casa_data_root)):
            if value:
                args += [flag, value]
        return args + (["--keep-ms"] if keep_ms else []) + (["--overwrite"] if overwrite else [])

    runner: Callable[[list], dict[str, str | None]] = run_here
    if backend == Backend.slurm and len(tasks) > 1:
        from ..services.noise.batch import run_on_slurm

        slurm = SlurmOptions(
            queue=slurm_queue,
            project=slurm_project,
            walltime=slurm_walltime,
            memory=slurm_memory,
            n_workers=slurm_workers,
            scheduler_host=scheduler_host,
            scheduler_interface=scheduler_interface,
        )
        runner = partial(
            run_on_slurm,
            out=out,
            step="validate",
            command=window_command,
            backend_kwargs={"node_cores": num_cores + 1, **slurm.backend_kwargs()},
            cores=num_cores,
            timeout=task_timeout,
            extra_args=[],
            echo=typer.echo,
        )
    todo = [t for t in tasks if overwrite or not v.done(t, out, noise_model)]
    failures = run_tasks(todo, out, runner, step="validate", overwrite=True, echo=typer.echo)
    if judge:
        _print_comparison(v.judge(read_windows(windows), out, noise_model, measured, rules))
    if _failed(failures, todo, allow_partial):
        raise typer.Exit(code=1)


def _validate_inputs(
    model: str, windows: Path, metadata: Path, criteria: Path | None, name: str | None
) -> tuple[Any, list, dict, Any]:
    """noise validate's model, windows, archive rows and criteria, checked."""
    from ..services.noise import validate as v
    from ..services.noise.batch import read_windows
    from ..services.noise.compare import Criteria
    from ..services.noise_model import resolve_noise_model

    try:
        noise_model = resolve_noise_model(model)
        tasks = read_windows(windows)
        rows = v.metadata_rows(metadata)
        rules = Criteria.model_validate_json(criteria.read_text()) if criteria else None
    except (OSError, ValueError) as exc:
        usage_error(str(exc))
    if noise_model is None:
        usage_error("--model: a model, not none")
    if name is not None:
        tasks = [t for t in tasks if t.name == name]
        if not tasks:
            usage_error(f"--name: no window {name!r} in {windows}")
    missing = [t.name for t in tasks if t.name not in rows]
    if missing:
        usage_error(f"--metadata: no row for {', '.join(missing[:5])}")
    return noise_model, tasks, rows, rules


@noise_app.command("compare")
def compare_windows(
    model: str = typer.Option(
        ..., "--model", help="The model validated.", rich_help_panel="Inputs"
    ),
    windows: Path = typer.Option(
        ..., "--windows", help="The windows table.", rich_help_panel="Inputs"
    ),
    measured: Path = typer.Option(
        ...,
        "--measured",
        help="noise extract's output for the real windows.",
        rich_help_panel="Inputs",
    ),
    out: Path = typer.Option(
        ..., "--out", help="noise validate's output.", rich_help_panel="Inputs"
    ),
    criteria: Path | None = typer.Option(
        None, "--criteria", help="A JSON file of comparison criteria.", rich_help_panel="Inputs"
    ),
) -> None:
    """Compare the windows noise validate simulated with the real ones
    (comparison.json): per metric the median ratio or difference, its range,
    Wilcoxon p, and whether the model passes the criteria."""
    from ..services.noise import validate as v
    from ..services.noise.batch import read_windows
    from ..services.noise.compare import Criteria
    from ..services.noise_model import resolve_noise_model

    try:
        noise_model = resolve_noise_model(model)
        tasks = read_windows(windows)
        rules = Criteria.model_validate_json(criteria.read_text()) if criteria else None
    except (OSError, ValueError) as exc:
        usage_error(str(exc))
    if noise_model is None:
        usage_error("--model: a model, not none")
    comparison = v.judge(
        tasks, out.expanduser().resolve(), noise_model, measured.expanduser().resolve(), rules
    )
    if comparison.windows == 0:
        usage_error(
            f"no validated windows of {noise_model.label()} under {out}; run noise validate first"
        )
    _print_comparison(comparison)
    if not comparison.passed:
        raise typer.Exit(code=1)


def _print_comparison(comparison) -> None:
    typer.echo(f"{comparison.model.get('label')}: {comparison.windows} window(s) compared")
    for metric, s in comparison.metrics.items():
        within = f", {s.within_10pct} within 10%" if s.within_10pct is not None else ""
        p = f", p {s.p:.2g}" if s.p is not None else ""
        typer.echo(
            f"  {metric:28s} {s.kind:10s} {s.median:+.4g} ({s.q16:+.3g}..{s.q84:+.3g}){p}{within}"
        )
    for check in comparison.checks:
        typer.echo(f"  [{'pass' if check.passed else 'FAIL'}] {check.name}: {check.value}")
    typer.echo("PASSED" if comparison.passed else "NOT PASSED")


@noise_app.command("promote")
def promote(
    candidate: Path = typer.Argument(
        ..., help="The candidate model (noise aggregate's, or any file)."
    ),
    comparison: Path = typer.Option(
        ..., "--comparison", help="noise validate's comparison.json of that very model."
    ),
    name: str | None = typer.Option(
        None, "--name", help="The model it releases (default: the candidate's, less -candidate)."
    ),
    models_dir: Path | None = typer.Option(
        None, "--models-dir", help="Where released models live (default: this checkout's)."
    ),
) -> None:
    """Write a validated candidate as the next version of a released model
    (models/<name>-<n+1>.json), refused unless its comparison passed. The pull
    request that adds the file, and with it moves the model to the new
    version, stays with a person."""
    from ..services.noise.compare import Comparison
    from ..services.noise.promote import MODELS_DIR
    from ..services.noise.promote import promote as release
    from ..services.noise_model import NoiseModel

    try:
        model = NoiseModel.model_validate_json(candidate.read_text())
        result = Comparison.model_validate_json(comparison.read_text())
    except (OSError, ValueError) as exc:
        usage_error(str(exc))
    try:
        path = release(model, result, name=name, models_dir=models_dir or MODELS_DIR)
    except ValueError as exc:
        fail(str(exc))
    released = NoiseModel.model_validate_json(path.read_text())
    typer.echo(f"{released.label()} written to {path}")
    typer.echo(
        "Next: add it in a pull request; update the tests that pin the model's values, "
        "regenerate docs/noise-model.schema.json if the schema changed, and label the PR "
        "golden:supersede if the released default changes what runs produce."
    )
