"""Noise models: what is released, what each component does, the file schema (MADIV-28).

``noise show`` lists the released models, or prints one (a released name, a
candidate file) with the components it applies; ``noise components`` lists
the components a model can hold and their parameters; ``noise schema``
writes the JSON Schema model files validate against. ``noise extract``
measures real windows' noise, one window or a table of them, on Slurm.
"""

from __future__ import annotations

import json
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
        help="A CSV of windows (columns ms, spw; optional field, name, syscal_ms, "
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
    if failures:
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
    if failures:
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
