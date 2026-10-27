"""What a job is: its command, resources, owner and state (web console plan, J1.1)."""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# Bump when the manifest's fields change incompatibly; readers refuse newer.
MANIFEST_SCHEMA = 1

# Where a job runs: a Slurm batch job, or a process beside the server (for
# the commands that need the internet, on sites whose compute nodes have none).
EXECUTORS = ("slurm", "server")


class JobState(StrEnum):
    """The life of a job, as the UI shows it."""

    PENDING = "pending"  # written, not submitted (waits on a parent step)
    QUEUED = "queued"  # submitted, not started
    RUNNING = "running"  # the wrapper reported its start
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"  # ended without the wrapper reporting why (node failure, OOM kill)

    @property
    def finished(self) -> bool:
        return self in _FINISHED


_FINISHED = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.LOST}

# Crockford base32, as ULIDs use: no I, L, O, U, so ids read back unambiguously.
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_JOB_ID = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")


def new_job_id(now_ms: int | None = None, entropy: bytes | None = None) -> str:
    """A ULID: 48 bits of milliseconds then 80 random bits, 26 characters.

    Ids sort by creation time as text, so a folder listing is in order, and
    they are unique across users without coordination.
    """
    ms = int(time.time() * 1000) if now_ms is None else now_ms
    rand = os.urandom(10) if entropy is None else entropy
    if not 0 <= ms < 1 << 48 or len(rand) != 10:
        raise ValueError("a job id needs a 48-bit time and 10 random bytes")
    value = (ms << 80) | int.from_bytes(rand, "big")
    return "".join(_ULID_ALPHABET[(value >> shift) & 31] for shift in range(125, -1, -5))


def is_job_id(text: str) -> bool:
    return bool(_JOB_ID.fullmatch(text))


# The CLI's shape (almasim.cli._GROUPS): groups whose subcommand is part of
# the command's name, and single-command groups whose subcommand word users
# may type or leave out (`almasim metadata query ...` is `metadata`).
_MULTI_COMMAND_GROUPS = frozenset({"products", "visibilities", "image", "preview", "noise"})
_SINGLE_COMMAND_GROUPS = {"metadata": "query", "simulation": "run", "predict": "ms-from-image"}


# In an argument, the run's own folder, filled in when the job runs: a
# command can write its results there (`--save-csv {run_dir}/results.csv`)
# before the folder exists. The manifest keeps the placeholder, so running it
# again writes into the new run's folder.
RUN_DIR = "{run_dir}"


def expand_run_dir(argv: list[str], run_dir: str | Path) -> list[str]:
    """``argv`` with ``{run_dir}`` replaced by the run folder."""
    return [arg.replace(RUN_DIR, str(run_dir)) for arg in argv]


def command_name(argv: list[str]) -> str:
    """The command an ``almasim`` argv runs, as profiles and the policy name it.

    ``["image", "image-from-ms", "p.csv"]`` is ``image image-from-ms``;
    ``["metadata", "query", "--band", "3"]`` and ``["metadata", "--band", "3"]``
    are ``metadata``; ``["clean", "ms.ms"]`` is ``clean``.
    """
    if not argv or argv[0].startswith("-"):
        return ""
    group = argv[0]
    if group in _MULTI_COMMAND_GROUPS and len(argv) > 1 and not argv[1].startswith("-"):
        return f"{group} {argv[1]}"
    return group


class ResourcesError(ValueError):
    """A resource request that Slurm would reject or that could inject directives."""


# Values end up in #SBATCH lines, so each must match a pattern WHOLE
# (fullmatch: `$` alone also matches before a trailing newline).
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}")
_TIME = re.compile(r"(?:\d{1,3}-)?\d{1,3}:\d{2}:\d{2}")
_MEMORY = re.compile(r"\d{1,7}[KMGT]?")


@dataclass(frozen=True)
class Resources:
    """What a job asks Slurm for. A server-executor job ignores all but ``cpus``.

    ``memory`` uses Slurm's own form (``16G``, ``500M``); ``time`` is a Slurm
    time limit (``02:00:00``, ``2-00:00:00``).
    """

    partition: str | None = None
    account: str | None = None
    cpus: int = 1
    memory: str = "4G"
    time: str = "02:00:00"

    def __post_init__(self) -> None:
        for name, pattern in (("partition", _NAME), ("account", _NAME)):
            value = getattr(self, name)
            if value is not None and not pattern.fullmatch(value):
                raise ResourcesError(f"{name} {value!r} is not a valid Slurm name")
        if (
            isinstance(self.cpus, bool)
            or not isinstance(self.cpus, int)
            or not 1 <= self.cpus <= 1024
        ):
            raise ResourcesError(f"cpus must be an integer from 1 to 1024, not {self.cpus!r}")
        if not _MEMORY.fullmatch(self.memory):
            raise ResourcesError(f"memory {self.memory!r} must look like 16G or 500M")
        if not _TIME.fullmatch(self.time):
            raise ResourcesError(f"time {self.time!r} must look like 02:00:00 or 2-00:00:00")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Resources:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Job:
    """One run of one ``almasim`` command; the content of its ``job.json``.

    ``argv`` is the command line after ``almasim`` (``["image",
    "image-from-ms", "params.csv", "out", "--cube"]``); ``params`` keeps the
    form values it was built from, so "edit and run again" can refill a form.
    """

    id: str
    owner: str
    argv: list[str]
    executor: str
    resources: Resources = field(default_factory=Resources)
    params: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    title: str = ""
    parents: list[str] = field(default_factory=list)
    slurm_job_id: str | None = None
    state: JobState = JobState.PENDING
    # Why the console set the state itself (e.g. "an earlier step did not
    # succeed"); the wrapper's and Slurm's reasons are not stored here.
    reason: str = ""
    schema: int = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if not is_job_id(self.id):
            raise ValueError(f"job id {self.id!r} is not a 26-character ULID")
        if self.executor not in EXECUTORS:
            raise ValueError(f"executor must be one of {EXECUTORS}, not {self.executor!r}")
        if not self.argv or not all(isinstance(a, str) for a in self.argv):
            raise ValueError("argv must be a non-empty list of strings")
        self.state = JobState(self.state)

    @property
    def command(self) -> str:
        """The command's name (``image image-from-ms``, ``metadata``); see :func:`command_name`."""
        return command_name(self.argv)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["resources"] = self.resources.to_dict()
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        schema = data.get("schema", MANIFEST_SCHEMA)
        if not isinstance(schema, int) or schema > MANIFEST_SCHEMA:
            raise ValueError(f"manifest schema {schema!r} is newer than this almasim reads")
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        known["resources"] = Resources.from_dict(data.get("resources") or {})
        return cls(**known)
