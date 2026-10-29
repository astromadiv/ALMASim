"""How jobs run on this cluster: the site and user profiles (web console plan, J1.3).

A profile is TOML. The site profile (``ALMASIM_SITE_PROFILE``, kept with the
shared install) says what every user shares: the runs root, the Python that
runs jobs, WSClean, the CASA data, the partitions. The user profile
(``ALMASIM_PROFILE``, else ``~/.config/almasim/almasim.toml``) overrides any of
it. Tables merge key by key, the user's value winning; lists are replaced.

    runs_root = "/SKAO_data/cycle11/almasim-runs"
    roots = ["/SKAO_data/cycle11", "~"]
    executor = "slurm"  # where commands run unless their table says; "server" without Slurm

    [python]
    executable = "/home/me/.conda/envs/py312/bin/python"
    setup = ["export OPENBLAS_NUM_THREADS=1"]

    [tools]
    wsclean = "/SKAO_data/spack/.../bin/wsclean"
    casa_data = "/SKAO_data/cycle11/.casa-data"
    noise_models = "/SKAO_data/cycle11/almasim/noise-models"  # the team's own models

    [slurm]
    partition = "zwickyA-CPU"
    account = "ska"

    [slurm.partitions.zwickyA-CPU]
    cpus = 96
    memory = "770G"

    [commands."image image-from-ms"]
    executor = "slurm"
    cpus = 4
    memory = "16G"
    time = "2-00:00:00"
    args = ["--backend", "slurm"]

Unknown keys are errors, so a typo does not silently fall back to a default.
"""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .model import EXECUTORS, Resources, ResourcesError

SITE_ENV = "ALMASIM_SITE_PROFILE"
USER_ENV = "ALMASIM_PROFILE"
DEFAULT_USER_PROFILE = Path("~/.config/almasim/almasim.toml")
DEFAULT_RUNS_ROOT = Path("~/almasim-runs")


class ProfileError(ValueError):
    """A profile file that cannot be read or holds something it should not."""


@dataclass(frozen=True)
class Partition:
    """What one Slurm partition offers per node; limits for the resource panel."""

    name: str
    cpus: int | None = None
    memory: str | None = None
    max_time: str | None = None


@dataclass(frozen=True)
class CommandPolicy:
    """Where one command runs and what it asks for, by default."""

    executor: str | None = None
    partition: str | None = None
    cpus: int | None = None
    memory: str | None = None
    time: str | None = None
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Profile:
    runs_root: Path = DEFAULT_RUNS_ROOT
    roots: tuple[Path, ...] = (Path("~"),)
    executor: str | None = None  # every command's, unless its own table names one
    python: str = sys.executable
    setup: tuple[str, ...] = ()
    wsclean: str | None = None
    casa_data: Path | None = None
    noise_models: Path | None = None
    partition: str | None = None
    account: str | None = None
    partitions: Mapping[str, Partition] = field(default_factory=dict)
    commands: Mapping[str, CommandPolicy] = field(default_factory=dict)
    sources: tuple[Path, ...] = ()  # the files it was read from, site first
    # The same files as absolute paths, which a job gets in its environment: it
    # runs in its run folder, where a relative path no longer points at them.
    site_file: Path | None = None
    user_file: Path | None = None

    def policy(self, command: str) -> CommandPolicy:
        return self.commands.get(command, CommandPolicy())

    def resources_for(self, command: str, **overrides: Any) -> Resources:
        """The resources ``command`` asks for: defaults, then the profile, then ``overrides``."""
        policy = self.policy(command)
        values: dict[str, Any] = {"partition": self.partition, "account": self.account}
        for name in ("partition", "cpus", "memory", "time"):
            value = getattr(policy, name)
            if value is not None:
                values[name] = value
        values.update({k: v for k, v in overrides.items() if v is not None})
        return Resources(**values)


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(path)).expanduser()


def _merge(base: dict[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


_TOP = {"runs_root", "roots", "executor", "python", "tools", "slurm", "commands"}
_PYTHON = {"executable", "setup"}
_TOOLS = {"wsclean", "casa_data", "noise_models"}
_SLURM = {"partition", "account", "partitions"}
_PARTITION = {"cpus", "memory", "max_time"}
_COMMAND = {"executor", "partition", "cpus", "memory", "time", "args"}


def _check_keys(table: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ProfileError(f"{where}: unknown key(s) {', '.join(unknown)}")


def _strings(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ProfileError(f"{where} must be a list of strings")
    return tuple(value)


def _paths(data: Mapping[str, Any], where: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "runs_root" in data:
        out["runs_root"] = _expand(data["runs_root"])
    if "executor" in data:
        if data["executor"] not in EXECUTORS:
            raise ProfileError(f"{where}: executor must be one of {', '.join(EXECUTORS)}")
        out["executor"] = data["executor"]
    if "roots" in data:
        out["roots"] = tuple(_expand(p) for p in _strings(data["roots"], f"{where}: roots"))
    python = data.get("python", {})
    _check_keys(python, _PYTHON, f"{where} [python]")
    if "executable" in python:
        out["python"] = str(_expand(python["executable"]))
    if "setup" in python:
        out["setup"] = _strings(python["setup"], f"{where}: python.setup")
    tools = data.get("tools", {})
    _check_keys(tools, _TOOLS, f"{where} [tools]")
    if "wsclean" in tools:
        out["wsclean"] = str(_expand(tools["wsclean"]))
    if "casa_data" in tools:
        out["casa_data"] = _expand(tools["casa_data"])
    if "noise_models" in tools:
        out["noise_models"] = _expand(tools["noise_models"])
    return out


def _slurm(data: Mapping[str, Any], where: str) -> dict[str, Any]:
    slurm = data.get("slurm", {})
    _check_keys(slurm, _SLURM, f"{where} [slurm]")
    partitions: dict[str, Partition] = {}
    for name, table in slurm.get("partitions", {}).items():
        _check_keys(table, _PARTITION, f"{where} [slurm.partitions.{name}]")
        partitions[name] = Partition(name=name, **table)
    return {
        "partition": slurm.get("partition") or None,
        "account": slurm.get("account") or None,  # "" means none
        "partitions": partitions,
    }


def _commands(data: Mapping[str, Any], where: str) -> dict[str, CommandPolicy]:
    commands: dict[str, CommandPolicy] = {}
    for name, table in data.get("commands", {}).items():
        at = f'{where} [commands."{name}"]'
        _check_keys(table, _COMMAND, at)
        if table.get("executor") not in (None, *EXECUTORS):
            raise ProfileError(f"{at}: executor must be one of {', '.join(EXECUTORS)}")
        args = _strings(table.get("args", []), f"{at} args")
        commands[name] = CommandPolicy(**{**table, "args": args})
    return commands


def _build(data: Mapping[str, Any], sources: tuple[Path, ...]) -> Profile:
    where = " + ".join(str(s) for s in sources) or "profile"
    _check_keys(data, _TOP, where)
    profile = Profile(
        sources=sources,
        **_paths(data, where),
        **_slurm(data, where),
        commands=_commands(data, where),
    )
    # Every partition and command default must make valid Slurm resources.
    try:
        profile.resources_for("")
        for name in profile.commands:
            profile.resources_for(name)
    except (ResourcesError, TypeError) as exc:
        raise ProfileError(f"{where}: {exc}") from exc
    return profile


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except OSError as exc:
        raise ProfileError(f"{path}: {exc.strerror or exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"{path}: {exc}") from exc


def profile_paths(environ: Mapping[str, str] | None = None) -> tuple[Path | None, Path]:
    """(site profile or None, user profile) as the environment names them."""
    env = os.environ if environ is None else environ
    site = _expand(env[SITE_ENV]) if env.get(SITE_ENV) else None
    user = _expand(env[USER_ENV]) if env.get(USER_ENV) else DEFAULT_USER_PROFILE.expanduser()
    return site, user


def load_profile(
    site: str | os.PathLike[str] | None = None,
    user: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Profile:
    """The merged profile. Without arguments the files come from the environment.

    A named site profile must exist; the user profile is optional unless it
    was named explicitly (by argument or ``ALMASIM_PROFILE``).
    """
    env = os.environ if environ is None else environ
    env_site, env_user = profile_paths(env)
    site_path = Path(site).expanduser() if site is not None else env_site
    user_named = user is not None or bool(env.get(USER_ENV))
    user_path = Path(user).expanduser() if user is not None else env_user

    data: dict[str, Any] = {}
    sources: list[Path] = []
    if site_path is not None:
        data = _merge(data, _read(site_path))
        sources.append(site_path)
    user_file = None
    if user_path.is_file() or user_named:
        data = _merge(data, _read(user_path))
        sources.append(user_path)
        user_file = user_path.absolute()
    return replace(
        _build(data, tuple(sources)),
        site_file=site_path.absolute() if site_path is not None else None,
        user_file=user_file,
    )
