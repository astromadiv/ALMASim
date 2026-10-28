"""Chains of runs: each step starts when the one before it succeeds (web console plan, J1.8).

A pipeline is a list of steps, each an ``almasim`` command line; every step
depends on the one before it. The templates name the files the steps hand
on to each other inside one pipeline folder, so a chain can be resumed or
rerun step by step from the folder alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .profile import Profile
from .service import RunView, submit_command


@dataclass(frozen=True)
class Step:
    title: str
    argv: list[str]
    resources: Mapping[str, Any] = field(default_factory=dict)


def archive_to_images(
    folder: str | Path,
    query: Sequence[str],
    *,
    download: Sequence[str] = (),
    imaging: Sequence[str] = (),
) -> list[Step]:
    """Query the ALMA archive, download and calibrate, then image and check quality.

    ``query`` are ``almasim metadata`` options (``--science-keyword``,
    ``--band``…); ``download`` and ``imaging`` add options to those steps.
    Everything lands in ``folder``: metadata.csv, downloads/, ms/ (raw and
    calibrated measurement sets), params.csv, images/.
    """
    f = Path(folder)
    calibrated = f / "ms" / "calibrated_ms"
    return [
        Step("Query the archive", ["metadata", "--save-csv", str(f / "metadata.csv"), *query]),
        Step(
            "Download and calibrate",
            [
                "products",
                "download",
                "--metadata-csv",
                str(f / "metadata.csv"),
                "--destination",
                str(f / "downloads"),
                "--extract-tar",
                "--unpack-ms",
                "--generate-calibrated-visibilities",
                "--archive-output-root",
                str(f / "ms"),
                *download,
            ],
        ),
        Step(
            "Imaging parameters",
            ["image", "compute-parameters", str(calibrated), str(f / "params.csv")],
        ),
        Step(
            "Imaging",
            ["image", "image-from-ms", str(f / "params.csv"), str(f / "images"), *imaging],
        ),
        Step("Clean quality", ["image", "clean-quality", str(f / "images")]),
    ]


def simulate(
    folder: str | Path, metadata_csv: str | Path, options: Sequence[str] = ()
) -> list[Step]:
    """Simulate the observations in ``metadata_csv`` into ``folder``."""
    return [
        Step(
            "Simulate",
            [
                "simulation",
                "--metadata-path",
                str(metadata_csv),
                "--output-dir",
                str(folder),
                *options,
            ],
        )
    ]


def noise_model_from_windows(
    folder: str | Path,
    windows_csv: str | Path,
    metadata_csv: str | Path,
    *,
    base: str = "alma",
    extract: Sequence[str] = (),
    fit: Sequence[str] = (),
    validate: Sequence[str] = (),
) -> list[Step]:
    """Measure real windows' noise, fit their calibration residuals, reduce
    them into a candidate noise model and validate it against them.

    ``windows_csv`` is ``noise extract``'s windows table (with ``task_dir``
    and ``pwv`` columns for the fit and the receiver temperatures),
    ``metadata_csv`` each window's archive row with a ``name`` column;
    ``extract``, ``fit`` and ``validate`` add options to those steps. Everything
    lands in ``folder``: windows/, candidate.json, validation/.
    """
    f = Path(folder)
    windows, candidate = str(windows_csv), str(f / "candidate.json")
    return [
        Step(
            "Measure the windows' noise",
            ["noise", "extract", "--windows", windows, "--out", str(f), *extract],
        ),
        Step(
            "Fit their calibration residuals",
            ["noise", "fit", "--windows", windows, "--out", str(f), *fit],
        ),
        Step(
            "Reduce into a candidate model",
            ["noise", "aggregate", "--out", str(f), "--from", base, "--output", candidate],
        ),
        Step(
            "Validate the candidate",
            [
                "noise",
                "validate",
                "--model",
                candidate,
                "--windows",
                windows,
                "--metadata",
                str(metadata_csv),
                "--measured",
                str(f),
                "--out",
                str(f),
                *validate,
            ],
        ),
    ]


def submit_pipeline(
    steps: Sequence[Step],
    profile: Profile,
    *,
    title: str = "",
    after: Sequence[Path] = (),
) -> list[RunView]:
    """Submit ``steps`` as a chain; each run waits for the one before it."""
    runs: list[RunView] = []
    previous: Sequence[Path] = after
    for i, step in enumerate(steps, start=1):
        name = f"{title}: {step.title}" if title else step.title
        run = submit_command(
            step.argv,
            profile,
            resources=step.resources,
            title=name,
            params={"pipeline": title, "step": i, "steps": len(steps)},
            after=previous,
        )
        runs.append(run)
        previous = [run.run_dir]
    return runs
