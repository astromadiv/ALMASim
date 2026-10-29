"""Where a command runs, what it asks for, and what the job adds (web console plan, J1.6).

``plan_job`` turns an ``almasim`` command line into what to submit:

- the executor: Slurm, for every command. zwicky's compute nodes reach the
  ALMA archive, so queries and downloads run there too and nothing heavy
  runs on the login node. A site whose compute nodes are offline sets
  ``[commands."<name>"] executor = "server"`` for the commands in
  NETWORK_COMMANDS; a machine without Slurm (a laptop) sets
  ``executor = "server"`` at the top of its profile, for every command;
- the resources: the profile's defaults for the command, then the caller's;
- the command line, with what the cluster needs filled in when the caller
  did not give it: the Slurm backend for the commands that fan out (their
  job is then a small, long driver), the partition and account for their
  workers, the profile's WSClean and CASA data, and no CASA data download
  on compute nodes. Then the profile's own ``args`` for the command.

An option the caller gave is never changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .model import Resources, command_name
from .profile import Profile

# The commands that need the internet: on Slurm by default, on the server
# node only where a profile says so (compute nodes without internet).
NETWORK_COMMANDS = frozenset(
    {"metadata", "products resolve", "products download", "products executions"}
)

KNOWN_COMMANDS = frozenset(
    {
        "metadata",
        "products resolve",
        "products download",
        "products executions",
        "simulation",
        "visibilities extract",
        "visibilities unpack",
        "visibilities calibrate",
        "visibilities spw-overview",
        "visibilities flag-summary",
        "predict",
        "image compute-parameters",
        "image image-from-ms",
        "image clean-quality",
        "image pb-correct",
        "clean",
        "preview integrate",
        "preview wiener",
        "noise windows",
        "noise extract",
        "noise fit",
        "noise aggregate",
        "noise validate",
        "noise compare",
        "doctor",
    }
)

# Commands that fan out to Slurm workers with --backend slurm.
_FAN_OUT = frozenset(
    {
        "visibilities unpack",
        "visibilities calibrate",
        "image image-from-ms",
        "noise extract",
        "noise fit",
        "noise validate",
    }
)
# products download fans out only for its post-processing.
_DOWNLOAD_POSTPROCESSING = ("--extract-tar", "--unpack-ms", "--generate-calibrated-visibilities")
_CASA_DATA = frozenset(
    {
        "products download",
        "visibilities unpack",
        "visibilities calibrate",
        "image image-from-ms",
        "noise validate",
    }
)
_SKIP_CASA_UPDATE = frozenset(
    {"products download", "visibilities unpack", "visibilities calibrate"}
)
_WSCLEAN = frozenset({"image image-from-ms", "clean", "noise validate"})
_CONFIRMS = frozenset({"metadata", "products download"})


DEFAULT_EXECUTOR = "slurm"


class PolicyError(ValueError):
    """A command line the console will not run."""


@dataclass(frozen=True)
class JobPlan:
    command: str
    executor: str
    argv: list[str]
    resources: Resources


def has_option(argv: list[str], *flags: str) -> bool:
    """Whether ``argv`` gives any of ``flags`` (as ``--x`` or ``--x=value``)."""
    return any(arg == f or arg.startswith(f + "=") for arg in argv for f in flags)


def option_value(argv: list[str], *flags: str) -> str | None:
    """The value ``argv`` gives the first of ``flags`` it has, or None."""
    for i, arg in enumerate(argv):
        for flag in flags:
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1]
            if arg == flag and i + 1 < len(argv):
                return argv[i + 1]
    return None


def _fills(command: str, argv: list[str], executor: str, profile: Profile) -> list[str]:
    extra: list[str] = []
    fans_out = command in _FAN_OUT or (
        command == "products download" and has_option(argv, *_DOWNLOAD_POSTPROCESSING)
    )
    backend = option_value(argv, "--backend", "--postprocess-backend")
    # Workers on Slurm only for a job on Slurm: a server job (a laptop, a
    # cluster's login node) keeps the command's own default backend.
    if fans_out and backend is None and executor == "slurm":
        extra += ["--backend", "slurm"]
        backend = "slurm"
    if fans_out and backend == "slurm":
        if not has_option(argv, "--slurm-queue") and profile.partition:
            extra += ["--slurm-queue", profile.partition]
        if not has_option(argv, "--slurm-project") and profile.account:
            extra += ["--slurm-project", profile.account]
    if command == "predict" and not has_option(argv, "--slurm-queue") and profile.partition:
        extra += ["--slurm-queue", profile.partition]
    if command in _CASA_DATA and profile.casa_data and not has_option(argv, "--casa-data-root"):
        extra += ["--casa-data-root", str(profile.casa_data)]
    if (
        command in _SKIP_CASA_UPDATE
        and executor == "slurm"
        and not has_option(argv, "--skip-casa-data-update")
    ):
        extra.append("--skip-casa-data-update")  # compute nodes cannot download it
    if command in _WSCLEAN and profile.wsclean and not has_option(argv, "--wsclean-bin"):
        extra += ["--wsclean-bin", profile.wsclean]
    if command in _CONFIRMS and not has_option(argv, "--yes", "-y"):
        extra.append("--yes")  # it asks for confirmation; a job has no terminal to answer
    return extra


def plan_job(
    argv: list[str], profile: Profile, resources: Mapping[str, Any] | None = None
) -> JobPlan:
    """What to submit for ``almasim <argv>``; ``resources`` override the profile's."""
    if not argv or not all(isinstance(a, str) for a in argv):
        raise PolicyError("give the command line as a list of strings")
    command = command_name(argv)
    if command not in KNOWN_COMMANDS:
        raise PolicyError(f"unknown command {' '.join(argv[:2])!r}")
    policy = profile.policy(command)
    executor = policy.executor or profile.executor or DEFAULT_EXECUTOR
    extra = [*_fills(command, argv, executor, profile), *policy.args]
    # Options after a "--" would be taken as positional arguments.
    cut = argv.index("--") if "--" in argv else len(argv)
    full = [*argv[:cut], *extra, *argv[cut:]]
    return JobPlan(
        command=command,
        executor=executor,
        argv=full,
        resources=profile.resources_for(command, **dict(resources or {})),
    )
