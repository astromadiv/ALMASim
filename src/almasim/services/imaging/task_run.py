"""The runner an imaging task's steps share (MADIV-28 T4.2)."""

from __future__ import annotations

import shlex
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from astropy.io import fits

from almasim.services.imaging.markers import write_imaging_failure_marker
from almasim.services.imaging.provenance import Provenance, provenance_header
from almasim.services.imaging.wsclean.runner import _run_with_retries


@dataclass
class TaskRun:
    """Where one imaging task writes, and how its WSClean commands run and fail.

    :meth:`run` retries WSClean deaths by signal up to ``retries`` times and
    keeps the last lines of output in ``tail``; :meth:`fail` writes the
    task's failure marker and returns the error to raise. ``history`` lists
    the steps that ran, each WSClean command among them, and :meth:`header`
    gives them, with ``provenance``, as the products' FITS cards.
    """

    out: Path
    log_path: Path
    ms_path: str
    spw: int
    env: dict[str, str] = field(default_factory=dict)
    timeout: float | None = None
    retries: int = 0
    tail: deque[str] = field(default_factory=lambda: deque(maxlen=60))
    provenance: Provenance | None = None
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.retries = max(int(self.retries), 0)
        if self.provenance is None:
            self.provenance = Provenance.for_ms([self.ms_path], spw=self.spw)

    def run(self, cmd: list[str], log: IO[str]) -> int:
        """Run one WSClean command, writing its output to ``log``; the attempts it took."""
        attempts = _run_with_retries(
            cmd, self.out, log, self.env, self.timeout, self.tail, self.fail, self.retries
        )
        # The task folder's paths relative to it, as the command runs there.
        folder = str(self.out)
        relative = [
            "." if part == folder else part.removeprefix(f"{folder}/") for part in map(str, cmd)
        ]
        self.history.append(shlex.join(relative))
        return attempts

    def note(self, step: str) -> None:
        """Record a step that is not a WSClean command."""
        self.history.append(step)

    def header(self) -> fits.Header:
        """The provenance cards and the steps so far, for the products' headers."""
        return provenance_header(self.provenance or Provenance(), self.history)

    def fail(self, error: str, returncode: int | None = None) -> RuntimeError:
        write_imaging_failure_marker(
            self.out,
            ms_path=self.ms_path,
            spw=self.spw,
            error=error,
            log_path=self.log_path,
            returncode=returncode,
        )
        return RuntimeError(f"{error}\nFull log: {self.log_path}")
