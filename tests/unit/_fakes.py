"""Fakes shared by the unit tests (MADIV-28 T0.5).

Helpers used by several test modules live here instead of being copied into
each of them.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from unittest.mock import MagicMock

import astropy.units as U
import numpy as np
from martini import DataCube


def make_datacube(n_px: int = 16, n_chan: int = 8) -> DataCube:
    """A small martini DataCube centred at 100 GHz with 0.1 arcsec pixels."""
    return DataCube(
        n_px_x=n_px,
        n_px_y=n_px,
        n_channels=n_chan,
        px_size=0.1 * U.arcsec,
        channel_width=0.1 * U.GHz,
        spectral_centre=100.0 * U.GHz,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    )


class FakeTable:
    """The subset of ``casacore.tables.table`` the imaging and CLI code use."""

    def __init__(self, columns: dict[str, np.ndarray]):
        self.columns = columns
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def colnames(self):
        return list(self.columns)

    def nrows(self):
        return len(next(iter(self.columns.values())))

    def iscelldefined(self, name, row):
        return name in self.columns

    def getcell(self, name, row):
        return self.columns[name][row]

    def getcol(self, name, startrow=0, nrow=-1, rowincr=1):
        column = self.columns[name]
        stop = len(column) if nrow < 0 else startrow + nrow * rowincr
        return column[startrow:stop:rowincr]

    @contextmanager
    def query(self, text):
        """The rows a TaQL selection of ``A IN [..]``, ``A != B`` and ``A == n``
        clauses joined by AND keeps."""
        keep = np.ones(self.nrows(), dtype=bool)
        for clause in (c.strip() for c in text.split(" AND ")):
            if m := re.fullmatch(r"(\w+) IN \[([\d,]*)\]", clause):
                keep &= np.isin(self.columns[m[1]], [int(x) for x in m[2].split(",") if x])
            elif m := re.fullmatch(r"(\w+) != (\w+)", clause):
                keep &= self.columns[m[1]] != self.columns[m[2]]
            elif m := re.fullmatch(r"(\w+) == (\d+)", clause):
                keep &= self.columns[m[1]] == int(m[2])
            else:
                raise ValueError(f"unsupported clause {clause!r}")
        yield FakeTable({k: v[keep] for k, v in self.columns.items()})

    def putcol(self, name, value, startrow=0, nrow=-1):
        self.columns[name][startrow : startrow + len(value)] = value

    def getcoldesc(self, name):
        return {"source": name}

    def addcols(self, desc):
        self.columns[desc["name"]] = np.zeros_like(self.columns[desc["source"]])

    def removecols(self, name):
        del self.columns[name]

    def close(self):
        self.closed = True


class FakeProcess:
    """A ``subprocess.Popen`` that prints ``lines`` and exits with ``returncode``."""

    def __init__(self, lines=(), returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode

    def wait(self):
        return self.returncode


def fake_popen(monkeypatch, lines=(), returncode=0) -> list:
    """Replace ``subprocess.Popen``; returns the ``(cmd, kwargs)`` of every call."""
    calls: list = []

    def popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return FakeProcess(lines, returncode)

    monkeypatch.setattr("subprocess.Popen", popen)
    return calls


def mock_dask_client(n_px: int, n_chan: int) -> MagicMock:
    """A Dask client stand-in: compute passes tasks through, gather returns a random cube."""
    client = MagicMock()
    client.compute.side_effect = lambda tasks: tasks
    client.gather.side_effect = lambda futures: [
        np.random.rand(n_chan, n_px, n_px).astype(np.float32)
    ]
    return client


def patch_everywhere(monkeypatch, module, name, value):
    """Patch ``module.name`` and every almasim module that imported the same object.

    archive_imaging re-exports functions that now live in imaging submodules;
    a patch only takes effect where the caller looks the name up, so it is
    applied in each almasim module holding that object.
    """
    import sys

    original = getattr(module, name)
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("almasim") and (
            getattr(mod, name, None) is original
        ):
            monkeypatch.setattr(mod, name, value)


class FakeTask:
    """A TaskRun stand-in: ``run(cmd, log)`` calls ``run(cmd)``; ``fail`` builds the error."""

    def __init__(self, out, run=lambda cmd: 1, fail=RuntimeError, timeout=None):
        self.out = out
        self._run = run
        self.fail = fail
        self.timeout = timeout

    def run(self, cmd, log):
        return self._run(cmd)

    def header(self):
        from astropy.io import fits

        return fits.Header()


def flat_run_kwargs(imaging, execution, **kwargs):
    """What a fake ``run_imaging_tasks`` received, under the old keyword names (MADIV-28 T4.1)."""
    flat = {
        "backend_kind": execution.backend,
        "cores_per_task": execution.cores_per_task,
        "node_cores": execution.node_cores,
        "queue": execution.queue,
        "project": execution.project,
        "walltime": execution.walltime,
        "memory": execution.memory,
        "n_jobs": execution.n_jobs,
        "scheduler_host": execution.scheduler_host,
        "scheduler_interface": execution.scheduler_interface,
        "task_timeout": execution.task_timeout,
        "task_retries": execution.task_retries,
        "continue_on_error": execution.continue_on_error,
        "scratch_root": execution.scratch_root,
        "pbcor": imaging.pbcor,
        **imaging.task_kwargs(kwargs.get("casa_data")),
        "imaging": imaging,
        "execution": execution,
    }
    flat.update(kwargs)
    return flat
