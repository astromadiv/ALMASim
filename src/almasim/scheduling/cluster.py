"""Singleton utilities for running Dask on top of Slurm.

This module provides:
1. A singleton wrapper around a ``dask_jobqueue.SLURMCluster`` and
   ``dask.distributed.Client``.
2. A top-level helper function that can be submitted with ``client.submit``
   to execute shell subcommands and collect outputs.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shlex
import socket
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

from almasim.jobs.rundir import RUN_DIR_ENV, DashboardAddress, clear_dashboard, write_dashboard
from almasim.services.paths import UnsafePathError, safe_join

logger = logging.getLogger(__name__)

try:
    from dask.distributed import Client
    from dask_jobqueue.slurm import SLURMCluster

    SLURM_DASK_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    Client = None
    SLURMCluster = None
    SLURM_DASK_AVAILABLE = False


# What a dask worker logs when it cannot open a connection to the scheduler.
_SCHEDULER_TIMEOUT = re.compile(r"Timed out trying to connect to (?P<address>tcp://\S+)")


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(socket.gethostbyname(host)).is_loopback
    except (OSError, ValueError):
        return False


def _partition_node_address(queue: str) -> str | None:
    """IP address of one node of ``queue`` as Slurm names it, or None without Slurm."""
    try:
        listing = subprocess.run(
            ["sinfo", "-h", "-N", "-p", queue, "-o", "%o"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for name in listing.split():
        try:
            return socket.gethostbyname(name)
        except OSError:
            continue
    return None


def _source_address_towards(address: str) -> str | None:
    """Local address the kernel would send from to reach ``address`` (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((address, 9))
            return probe.getsockname()[0]
    except OSError:
        return None


def default_scheduler_host(queue: str) -> str | None:
    """Address the workers of ``queue`` should reach the scheduler on.

    The submit host's own name is not reliable: on a compute node
    ``compute-node11`` resolves to 127.0.1.1, so the scheduler listened on
    loopback and the worker on compute-node47 timed out (job 5266,
    2026-10-17). The address the kernel routes to the partition's nodes is
    the one they can answer. ``HOSTNAME`` is the fallback without Slurm,
    unless it too resolves to loopback (then Dask picks).
    """
    node = _partition_node_address(queue)
    source = _source_address_towards(node) if node else None
    if source and not ipaddress.ip_address(source).is_loopback:
        return source
    hostname = os.environ.get("HOSTNAME")
    if hostname and not _is_loopback(hostname):
        return hostname
    return None


@dataclass(frozen=True)
class SubcommandResult:
    """Result produced by ``run_subcommand``."""

    stdout: str
    stderr: str
    returncode: int


def _call_with_kwargs(func: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    """Worker-side trampoline so task kwargs never collide with ``Client.submit``'s."""
    return func(**dict(kwargs))


def run_subcommand(
    command: str | Sequence[str],
    cores: int,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    shell: bool = False,
) -> SubcommandResult:
    """Execute a subcommand on a worker and return stdout/stderr/return code.

    Parameters
    ----------
    command : str | Sequence[str]
        Command to execute.
    cores : int
        Number of cores requested by this task. Must be less than the number
        of cores available on the worker node.
    cwd : str | None
        Optional working directory.
    env : Mapping[str, str] | None
        Optional environment variables to merge with the current environment.
    timeout : float | None
        Optional timeout in seconds.
    shell : bool
        Whether to execute through the shell.
    """
    if cores <= 0:
        raise ValueError("cores must be greater than 0")

    node_cores = os.cpu_count() or 1
    if cores >= node_cores:
        raise ValueError(f"cores ({cores}) must be less than node cores ({node_cores})")

    command_env = os.environ.copy()
    if env:
        command_env.update(dict(env))

    # Prioritize system libraries (glibc, etc.) over spack versions to avoid version conflicts.
    # Some spack builds may be compiled for newer GLIBC versions than the system provides.
    ld_library_path = "/lib64:/usr/lib64:/usr/local/lib64:/lib:/usr/lib:/usr/local/lib"
    existing_ld = command_env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        # Prepend system paths before existing paths so they take priority.
        ld_library_path = f"{ld_library_path}:{existing_ld}"
    command_env["LD_LIBRARY_PATH"] = ld_library_path

    # Keep threaded libraries aligned with requested task cores.
    for var_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        command_env[var_name] = str(cores)

    run_command: str | Sequence[str]
    if shell:
        run_command = command if isinstance(command, str) else " ".join(command)
    else:
        run_command = shlex.split(command) if isinstance(command, str) else command

    try:
        completed = subprocess.run(
            run_command,
            cwd=cwd,
            env=command_env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            shell=shell,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or "")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or "")
        stderr = f"{stderr}\nCommand timed out after {timeout} seconds".strip()
        return SubcommandResult(
            stdout=stdout,
            stderr=stderr,
            returncode=124,
        )
    return SubcommandResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


class SlurmClusterBusyError(RuntimeError):
    """The shared cluster is running with other settings (MADIV-28 W0.3)."""


class SlurmDaskClusterSingleton:
    """Singleton manager for a Slurm-backed Dask cluster."""

    _instance: ClassVar[SlurmDaskClusterSingleton | None] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()
    # How many get_instance() callers still use the cluster; the last
    # release_instance() closes it (MADIV-28 W0.3).
    _holders: ClassVar[int] = 0

    @staticmethod
    def _resolve_safe_log_directory(raw_log_directory: Any) -> str:
        """Resolve and validate that worker logs stay in a directory under the user's home.

        :func:`~almasim.services.paths.safe_join` resolves ``..`` segments and
        symlinks before comparing with the home directory plus a separator, so
        neither ``~/../etc`` nor a sibling such as ``/home/user2`` for
        ``/home/user`` passes. A relative path is taken from the home directory,
        and the home directory itself is refused.
        """
        if not isinstance(raw_log_directory, str):
            raise ValueError("log_directory must be a string")
        if not raw_log_directory.strip():
            raise ValueError("log_directory must not be empty")
        outside = ValueError("log_directory must be within the user's home directory")
        try:
            log_directory = safe_join(Path.home(), raw_log_directory)
        except UnsafePathError as exc:
            raise outside from exc
        if log_directory == Path(os.path.realpath(Path.home())):
            raise outside
        return str(log_directory)

    def __init__(  # noqa: C901
        self,
        queue: str,
        node_cores: int,
        memory: str,
        walltime: str,
        n_jobs: int,
        project: str | None = None,
        scheduler_host: str | None = None,
        scheduler_interface: str | None = None,
        **cluster_kwargs: Any,
    ) -> None:
        if not SLURM_DASK_AVAILABLE:
            raise ImportError(
                "dask-jobqueue is not installed. Install it with: pip install dask-jobqueue"
            )
        if node_cores <= 1:
            raise ValueError("node_cores must be greater than 1")
        if n_jobs <= 0:
            raise ValueError("n_jobs must be greater than 0")
        if Client is None or SLURMCluster is None:
            raise RuntimeError("Dask Slurm dependencies are not available")

        submit_hostname = os.environ.get("HOSTNAME")
        # Dask refuses a host together with an interface, so the interface,
        # when given, chooses the address on its own.
        effective_scheduler_host = scheduler_host or (
            None if scheduler_interface else default_scheduler_host(queue)
        )

        scheduler_options = dict(cluster_kwargs.pop("scheduler_options", {}))
        if effective_scheduler_host and "host" not in scheduler_options:
            scheduler_options["host"] = effective_scheduler_host
        # Any free port: two drivers on one node both asked for 8787, and the
        # console finds the port in the run folder anyway (_record_dashboard).
        scheduler_options.setdefault("dashboard_address", ":0")

        job_script_prologue = list(cluster_kwargs.pop("job_script_prologue", []))
        if submit_hostname and not any(
            line.startswith("export HOSTNAME=") for line in job_script_prologue
        ):
            # Propagate submit-node hostname to workers when requested by HPC setup.
            job_script_prologue.append(f"export HOSTNAME={shlex.quote(submit_hostname)}")

        # Ensure system GLIBC takes priority over spack versions to avoid incompatibilities.
        if not any("LD_LIBRARY_PATH" in line for line in job_script_prologue):
            ld_path = "/lib64:/usr/lib64:/usr/local/lib64:/lib:/usr/lib:/usr/local/lib"
            job_script_prologue.append(f'export LD_LIBRARY_PATH="{ld_path}:$LD_LIBRARY_PATH"')

        worker_extra_args = list(cluster_kwargs.pop("worker_extra_args", []))
        if "--resources" not in worker_extra_args:
            worker_extra_args.extend(["--resources", f"CPU={node_cores}"])

        self.queue = queue
        self.node_cores = node_cores
        self.memory = memory
        self.walltime = walltime
        self.n_jobs = n_jobs
        self.project = project

        default_log_dir = os.path.join(os.path.expanduser("~"), "dask-worker-logs")
        raw_log_directory = cluster_kwargs.pop("log_directory", default_log_dir)
        log_directory = self._resolve_safe_log_directory(raw_log_directory)
        os.makedirs(log_directory, exist_ok=True)
        self.log_directory = log_directory

        slurm_kwargs = {
            "queue": queue,
            "cores": node_cores,
            "processes": 1,
            "memory": memory,
            "walltime": walltime,
            "worker_extra_args": worker_extra_args,
            "job_script_prologue": job_script_prologue,
            "scheduler_options": scheduler_options,
            "log_directory": log_directory,
            **cluster_kwargs,
        }
        if scheduler_interface:
            slurm_kwargs.setdefault("interface", scheduler_interface)
        if project:
            slurm_kwargs["project"] = project

        self.cluster = SLURMCluster(**slurm_kwargs)
        self.cluster.scale(n_jobs)
        self.client = Client(self.cluster)
        logger.info("Dask dashboard: %s", getattr(self.client, "dashboard_link", None))
        self._record_dashboard()
        logger.info("Slurm worker logs (dask-worker stdout/stderr): %s", log_directory)
        self._config_signature = (
            queue,
            node_cores,
            memory,
            walltime,
            n_jobs,
            project,
            scheduler_host,
            scheduler_interface,
        )

    def _record_dashboard(self) -> None:
        """Note the dashboard's address in the job's run folder, for the console to relay.

        Only inside a console job (the wrapper sets ``$ALMASIM_RUN_DIR``) and
        only when the scheduler serves a dashboard (distributed needs bokeh).
        """
        run_dir = os.environ.get(RUN_DIR_ENV)
        if not run_dir:
            return
        port = self.cluster.scheduler_info.get("services", {}).get("dashboard")
        host = urlsplit(self.cluster.scheduler_address).hostname
        if not port or not host:
            return
        try:
            write_dashboard(run_dir, DashboardAddress(host=host, port=int(port)))
        except OSError as exc:
            logger.warning("Could not record the Dask dashboard in %s: %s", run_dir, exc)

    def unreachable_scheduler_report(self) -> str | None:
        """Why the worker jobs cannot reach the scheduler, from their logs, or None.

        A worker that cannot connect exits after 30 s and its Slurm job ends,
        so without this the driver waits the full worker timeout for nothing.
        """
        for job in list(getattr(self.cluster, "workers", {}).values()):
            job_id = getattr(job, "job_id", None)
            if not job_id:
                continue
            log_path = os.path.join(self.log_directory, f"dask-worker-{job_id}.err")
            try:
                with open(log_path, encoding="utf-8", errors="replace") as log:
                    match = _SCHEDULER_TIMEOUT.search(log.read())
            except OSError:
                continue
            if match:
                return (
                    f"Slurm worker job {job_id} could not reach the Dask scheduler at "
                    f"{match['address']} (see {log_path}). The compute nodes cannot connect "
                    "to this host: run the command from a compute node (e.g. inside an sbatch "
                    "job) or pass --scheduler-host / --scheduler-interface with an address "
                    "they can reach."
                )
        return None

    @classmethod
    def get_instance(
        cls,
        queue: str = "normal",
        node_cores: int = 32,
        memory: str = "64GB",
        walltime: str = "02:00:00",
        n_jobs: int = 1,
        project: str | None = None,
        scheduler_host: str | None = None,
        scheduler_interface: str | None = None,
        **cluster_kwargs: Any,
    ) -> SlurmDaskClusterSingleton:
        """Create (once) or return the singleton cluster manager instance."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(
                    queue=queue,
                    node_cores=node_cores,
                    memory=memory,
                    walltime=walltime,
                    n_jobs=n_jobs,
                    project=project,
                    scheduler_host=scheduler_host,
                    scheduler_interface=scheduler_interface,
                    **cluster_kwargs,
                )
                cls._holders = 1
                return cls._instance

            requested_signature = (
                queue,
                node_cores,
                memory,
                walltime,
                n_jobs,
                project,
                scheduler_host,
                scheduler_interface,
            )
            if cls._instance._config_signature != requested_signature:
                raise SlurmClusterBusyError(
                    "SlurmDaskClusterSingleton already initialized with different settings. "
                    "Call close_instance() first if you need a different configuration."
                )
            cls._holders += 1
            return cls._instance

    @classmethod
    def release_instance(cls) -> None:
        """Give back one get_instance() hold; the last one closes the cluster.

        Two simulations sharing the cluster used to close it under each
        other: the first to finish called close_instance().
        """
        with cls._instance_lock:
            if cls._instance is None:
                return
            cls._holders -= 1
            if cls._holders > 0:
                return
            cls._close_locked()

    @classmethod
    def close_instance(cls) -> None:
        """Close and clear the singleton cluster instance, whoever still holds it."""
        with cls._instance_lock:
            cls._close_locked()

    @classmethod
    def _close_locked(cls) -> None:
        if cls._instance is not None:
            cls._instance.client.close()
            cls._instance.cluster.close()
            cls._instance = None
            if os.environ.get(RUN_DIR_ENV):
                clear_dashboard(os.environ[RUN_DIR_ENV])
        cls._holders = 0

    def submit_callable(self, func: Callable[..., Any], *, cores: int, **kwargs: Any) -> Any:
        """Submit ``func(**kwargs)`` to a worker, reserving ``cores`` CPU resources.

        Same accounting as :meth:`submit_subcommand`, for tasks that need to
        run Python on the worker (a subprocess wrapper that streams a log and
        writes markers, say) instead of a bare shell command.
        """
        if cores <= 0:
            raise ValueError("cores must be greater than 0")
        if cores >= self.node_cores:
            raise ValueError(f"cores ({cores}) must be less than node_cores ({self.node_cores})")
        return self.client.submit(
            _call_with_kwargs, func, kwargs, resources={"CPU": cores}, pure=False
        )

    def submit_subcommand(
        self,
        command: str | Sequence[str],
        cores: int,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        shell: bool = False,
    ) -> Any:
        """Submit ``run_subcommand`` with worker CPU-resource constraints."""
        if cores <= 0:
            raise ValueError("cores must be greater than 0")
        if cores >= self.node_cores:
            raise ValueError(f"cores ({cores}) must be less than node_cores ({self.node_cores})")

        return self.client.submit(
            run_subcommand,
            command=command,
            cores=cores,
            cwd=cwd,
            env=env,
            timeout=timeout,
            shell=shell,
            resources={"CPU": cores},
        )
