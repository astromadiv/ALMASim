"""Options that several commands share (MADIV-28 T3.2), and one spelling for each (T3.9).

Each alias is an ``Annotated`` type carrying its ``typer.Option``; a command
declares ``slurm_queue: SlurmQueue = "normal"`` and keeps its own default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any

import typer

# Old spellings still accepted (MADIV-28 T3.9), and the one to use instead.
DEPRECATED_FLAGS = {
    "--postprocess-backend": "--backend",
    "--slurm-scheduler-host": "--scheduler-host",
    "--slurm-n-jobs": "--slurm-workers",
    "--output-root": "--output-dir",
    "--corruptions": "--noise-model",
}


def deprecated_flag_warnings(args: list[str]) -> list[str]:
    """A warning for every deprecated option spelling in ``args``."""
    warnings = []
    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in DEPRECATED_FLAGS:
            warnings.append(
                f"WARNING: {flag} is deprecated and will be removed; use {DEPRECATED_FLAGS[flag]}."
            )
    return warnings


class Backend(StrEnum):
    """Where a command's tasks run."""

    sync = "sync"
    slurm = "slurm"


PostprocessBackend = Annotated[
    Backend,
    typer.Option(
        "--backend",
        "--postprocess-backend",
        case_sensitive=False,
        help="Where the tasks run: sync (this process, one at a time) or slurm (Slurm jobs).",
        rich_help_panel="Cluster",
    ),
]
SlurmQueue = Annotated[
    str, typer.Option("--slurm-queue", help="Slurm queue/partition.", rich_help_panel="Cluster")
]
SlurmProject = Annotated[
    str | None,
    typer.Option("--slurm-project", help="Slurm project/account.", rich_help_panel="Cluster"),
]
SlurmWalltime = Annotated[
    str,
    typer.Option(
        "--slurm-walltime",
        help="Slurm walltime per worker job (HH:MM:SS).",
        rich_help_panel="Cluster",
    ),
]
SlurmMemory = Annotated[
    str, typer.Option("--slurm-memory", help="Memory per Slurm worker.", rich_help_panel="Cluster")
]
SlurmCores = Annotated[
    int,
    typer.Option("--slurm-cores", min=1, help="Cores per Slurm worker.", rich_help_panel="Cluster"),
]
SchedulerHost = Annotated[
    str | None,
    typer.Option(
        "--scheduler-host",
        "--slurm-scheduler-host",
        help="Scheduler host advertised to the workers (default: this host).",
        rich_help_panel="Cluster",
    ),
]
SchedulerInterface = Annotated[
    str | None,
    typer.Option(
        "--scheduler-interface",
        help="Scheduler/worker network interface (e.g. ib0, eth0).",
        rich_help_panel="Cluster",
    ),
]
TaskTimeout = Annotated[
    float,
    typer.Option(
        "--task-timeout",
        min=1,
        help="Timeout in seconds for each worker-side command.",
        rich_help_panel="Cluster",
    ),
]
MaxCoresPerNode = Annotated[
    int,
    typer.Option(
        "--max-cores-per-node",
        min=1,
        help="Cores per node, for the Slurm workers' resource accounting.",
        rich_help_panel="Cluster",
    ),
]


@dataclass(frozen=True)
class SlurmOptions:
    """The Slurm worker settings of a command, as compute-backend keyword arguments."""

    queue: str = "normal"
    project: str | None = None
    walltime: str = "02:00:00"
    memory: str = "16GB"
    n_workers: int = 1
    cores: int | None = None
    scheduler_host: str | None = None
    scheduler_interface: str | None = None

    def backend_kwargs(self) -> dict[str, Any]:
        """``create_backend("slurm", **kwargs)`` arguments; unset optional ones are left out."""
        kwargs: dict[str, Any] = {
            "queue": self.queue,
            "project": self.project,
            "walltime": self.walltime,
            "memory": self.memory,
            "n_workers": self.n_workers,
        }
        for name in ("cores", "scheduler_host", "scheduler_interface"):
            value = getattr(self, name)
            if value not in (None, ""):
                kwargs[name] = value
        return kwargs
