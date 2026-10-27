"""Many windows' noise, measured one window per task (MADIV-28).

A windows table (CSV) lists one window per row: ``ms`` and ``spw`` (needed),
``field``, ``name``, ``syscal_ms``, ``line_channels`` ("a~b;c") and
``task_log`` (an ALMASim imaging task's log, whose "lines in channels" gives
them), ``pwv`` (the observation's water vapour, mm, which noise aggregate
needs for the receiver temperatures), and for ``noise fit`` ``task_dir``
(the imaging task holding the window's clean model). Each window writes
``<out>/windows/<name>/window.json`` (and ``antenna.npz``) when extracted,
``fit.json`` when fitted; a window whose file exists for the same input is
skipped, so a run that stopped resumes where it was.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .measurements import WindowFit, WindowNoise

WINDOW_FILE = "window.json"
FIT_FILE = "fit.json"
STEPS = ("extract", "fit")
ARRAYS_FILE = "antenna.npz"
FAILURES_FILE = "failures.json"


@dataclass(frozen=True)
class WindowTask:
    """One window to measure."""

    ms: str
    spw: int
    name: str
    field: str | None = None
    syscal_ms: str | None = None
    line_channels: str | None = None
    task_log: str | None = None
    task_dir: str | None = None
    pwv: str | None = None

    def directory(self, out: Path) -> Path:
        return out / "windows" / self.name

    def measured(self, out: Path) -> WindowNoise | None:
        """Its extraction, if made for this input."""
        path = self.directory(out) / WINDOW_FILE
        if not path.is_file():
            return None
        try:
            window = WindowNoise.model_validate_json(path.read_text())
        except ValueError:
            return None
        same = (window.input.ms, window.input.spw, window.input.field) == (
            self.ms,
            self.spw,
            self.field,
        )
        return window if same else None

    def done(self, out: Path, step: str = "extract") -> bool:
        """``step`` made already, for this input."""
        if step == "extract":
            return self.measured(out) is not None
        path = self.directory(out) / FIT_FILE
        if not path.is_file() or self.measured(out) is None:
            return False
        try:
            WindowFit.model_validate_json(path.read_text())
        except ValueError:
            return False
        return path.stat().st_mtime >= (self.directory(out) / WINDOW_FILE).stat().st_mtime

    def arguments(self, step: str = "extract") -> list[str]:
        """This window as ``almasim noise <step>`` arguments (less ``--out``)."""
        args = [self.ms, "--spw", str(self.spw), "--name", self.name]
        options = [("--field", self.field)]
        if step == "extract":
            options += [
                ("--syscal-ms", self.syscal_ms),
                ("--line-channels", self.line_channels),
                ("--task-log", self.task_log),
                ("--pwv", self.pwv),
            ]
        else:
            options += [("--task-dir", self.task_dir)]
        for flag, value in options:
            if value:
                args += [flag, value]
        return args


def window_name(ms: str, spw: int, field: str | None) -> str:
    """A window's default name: the MS's uid, its spw and field."""
    stem = Path(ms).name
    for suffix in (".split.cal", ".split", ".ms"):
        stem = stem.removesuffix(suffix)
    name = f"{stem}_spw{spw}" + (f"_field{field}" if field not in (None, "") else "")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def read_windows(path: Path) -> list[WindowTask]:
    """The windows a table lists; a ValueError for a table without ms and spw."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"ms", "spw"} - set(table.columns)
    if missing:
        raise ValueError(f"{path}: needs column(s) {', '.join(sorted(missing))}")
    tasks = []
    for row in table.to_dict("records"):
        spw = int(row["spw"])
        field = row.get("field") or None
        tasks.append(
            WindowTask(
                ms=row["ms"],
                spw=spw,
                name=row.get("name") or window_name(row["ms"], spw, field),
                field=field,
                syscal_ms=row.get("syscal_ms") or None,
                line_channels=row.get("line_channels") or None,
                task_log=row.get("task_log") or None,
                task_dir=row.get("task_dir") or None,
                pwv=row.get("pwv") or None,
            )
        )
    names = [t.name for t in tasks]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"{path}: window names repeat: {', '.join(duplicates)}")
    return tasks


def measure_task(task: WindowTask, out: Path, *, max_visibilities: int = 40_000_000) -> Path:
    """Measure one window and write its files; the window.json path."""
    from .extract import extract_window, line_channels_from_log, parse_channels, window_channels

    nchan = window_channels(task.ms, task.spw)
    lines = None
    if task.line_channels:
        lines = parse_channels(task.line_channels, nchan)
    elif task.task_log:
        lines = line_channels_from_log(Path(task.task_log), nchan)
    record, arrays = extract_window(
        task.ms,
        task.spw,
        task.field,
        name=task.name,
        syscal_ms=task.syscal_ms,
        line_channels=lines,
        max_visibilities=max_visibilities,
        pwv_mm=float(task.pwv) if task.pwv else None,
    )
    directory = task.directory(out)
    directory.mkdir(parents=True, exist_ok=True)
    if arrays:
        tmp = directory / f".{ARRAYS_FILE}.tmp.npz"
        np.savez_compressed(tmp, antenna_g=arrays["antenna_g"], antenna_h=arrays["antenna_h"])
        os.replace(tmp, directory / ARRAYS_FILE)
    path = directory / WINDOW_FILE
    _write_atomic(path, record.model_dump_json(indent=2) + "\n")
    return path


def fit_task(task: WindowTask, out: Path, **options: Any) -> Path:
    """Fit one window's calibration residuals (its extraction first) and write fit.json."""
    from .fit import fit_window

    window = task.measured(out)
    if window is None:
        raise ValueError(f"{task.name}: no window.json for this input; run noise extract first")
    if not task.task_dir:
        raise ValueError(f"{task.name}: noise fit needs the window's imaging task (--task-dir)")
    fitted = fit_window(task.ms, task.spw, task.field, task.task_dir, window, **options)
    path = task.directory(out) / FIT_FILE
    _write_atomic(path, fitted.model_dump_json(indent=2) + "\n")
    return path


def run_tasks(
    tasks: list[WindowTask],
    out: Path,
    run: Callable[[list[WindowTask]], dict[str, str | None]],
    *,
    step: str = "extract",
    overwrite: bool = False,
    echo: Callable[[str], Any] = print,
) -> dict[str, str]:
    """Run ``step`` on the windows that need it with ``run`` (window name ->
    None on success, else the error); writes failures.json and returns the failures."""
    todo = [t for t in tasks if overwrite or not t.done(out, step)]
    if len(todo) < len(tasks):
        echo(f"{len(tasks) - len(todo)} window(s) already measured; {len(todo)} to go")
    results = run(todo) if todo else {}
    failures = {name: error for name, error in results.items() if error}
    out.mkdir(parents=True, exist_ok=True)
    name = FAILURES_FILE if step == "extract" else f"failures-{step}.json"
    _write_atomic(out / name, json.dumps(failures, indent=2, sort_keys=True) + "\n")
    return failures


def run_in_process(
    todo: list[WindowTask],
    *,
    out: Path,
    step: str = "extract",
    echo: Callable[[str], Any] = print,
    **options: Any,
) -> dict[str, str | None]:
    """Run ``step`` on the windows here, one after the other (``options``:
    measure_task's or fit_task's)."""
    results: dict[str, str | None] = {}
    for task in todo:
        try:
            path = (measure_task if step == "extract" else fit_task)(task, out, **options)
            echo(f"{task.name}: {path}")
            results[task.name] = None
        except Exception as exc:  # noqa: BLE001 - one window must not stop the others
            echo(f"{task.name}: FAILED {exc}")
            results[task.name] = f"{type(exc).__name__}: {exc}"
    return results


def run_on_slurm(
    todo: list[WindowTask],
    *,
    out: Path,
    backend_kwargs: dict[str, Any],
    cores: int,
    timeout: float | None,
    extra_args: list[str],
    step: str = "extract",
    echo: Callable[[str], Any] = print,
) -> dict[str, str | None]:
    """Run each window as its own ``almasim noise <step>`` on a Slurm worker:
    a window's memory and any crash of its readers stay in its task."""
    from ...jobs.wrapper import ALMASIM_COMMAND
    from ..compute import create_backend

    kwargs = {
        **backend_kwargs,
        "n_workers": min(int(backend_kwargs.get("n_workers", 1)), len(todo)),
    }
    results: dict[str, str | None] = {}
    with create_backend("slurm", **kwargs) as cluster:
        futures = [
            (
                task,
                cluster.submit_subcommand(
                    command=[
                        *ALMASIM_COMMAND,
                        "noise",
                        step,
                        *task.arguments(step),
                        "--out",
                        str(out),
                        *extra_args,
                    ],
                    cores=cores,
                    timeout=timeout,
                ),
            )
            for task in todo
        ]
        for task, future in futures:
            result = future.result()
            if result.returncode == 0:
                echo(f"{task.name}: done")
                results[task.name] = None
            else:
                echo(f"{task.name}: FAILED")
                results[task.name] = (result.stderr or "").strip()[-2000:] or "failed"
    return results


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
