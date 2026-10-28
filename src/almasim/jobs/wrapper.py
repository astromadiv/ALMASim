"""Run one job's command and report how it went (web console plan, J1.2).

    python -m almasim.jobs.wrapper <run dir>

This is the only thing a job's Slurm script, or the server executor, calls.
It reads the run folder's job.json, runs ``almasim <argv>`` (``{run_dir}`` in an
argument becomes the run folder, and ``$ALMASIM_RUN_DIR`` names it) with its
output in stdout.log and stderr.log, and writes status.json twice: when the command
starts (host, time) and when it ends (time, exit code, and the signal if it
was stopped). A job cancelled with ``scancel`` receives SIGTERM; the wrapper
passes it on to the command, waits for it, and records the signal, so the
run reads as cancelled rather than failed.
"""

from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType

from .model import expand_run_dir
from .rundir import RUN_DIR_ENV, RunStatus, read_manifest, write_status

STDOUT_NAME = "stdout.log"
STDERR_NAME = "stderr.log"

# The command a job runs, before its argv: this interpreter's almasim CLI, so
# a job uses the same environment as whatever submitted it.
ALMASIM_COMMAND: tuple[str, ...] = (sys.executable, "-m", "almasim.cli")
# Another command in its place, split as a shell would: the console's
# end-to-end tests run instant stand-ins for the real commands with it.
COMMAND_ENV = "ALMASIM_JOB_COMMAND"


def job_command(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The command jobs run: ``$ALMASIM_JOB_COMMAND`` if set, else the almasim CLI."""
    env = os.environ if environ is None else environ
    return tuple(shlex.split(env[COMMAND_ENV])) if env.get(COMMAND_ENV) else ALMASIM_COMMAND


_FORWARDED = (signal.SIGTERM, signal.SIGINT)


def run_job(run_dir: str | os.PathLike[str], command: Sequence[str] = ALMASIM_COMMAND) -> int:
    """Run the job in ``run_dir``; its exit code (128 + n when signal n stopped it)."""
    run_dir = Path(run_dir)
    job = read_manifest(run_dir)
    # Shared runs roots are group-writable: keep what the job writes so.
    os.umask(0o002)

    started = RunStatus(started_at=time.time(), host=socket.gethostname())
    write_status(run_dir, started)

    received: list[int] = []
    with (run_dir / STDOUT_NAME).open("ab") as out, (run_dir / STDERR_NAME).open("ab") as err:
        try:
            child = subprocess.Popen(
                [*command, *expand_run_dir(job.argv, run_dir.absolute())],
                cwd=run_dir,
                env={**os.environ, RUN_DIR_ENV: str(run_dir.absolute())},
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,  # its own process group, so a stop reaches all of it
            )
        except OSError as exc:
            # The command cannot start (missing, not executable): the run
            # fails with the shell's exit code instead of staying "running".
            err.write(f"almasim job: cannot start {command[0]}: {exc}\n".encode())
            err.flush()
            return _finish(run_dir, started, 127 if isinstance(exc, FileNotFoundError) else 126)

        def forward(signum: int, _frame: FrameType | None) -> None:
            received.append(signum)
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

        previous = {s: signal.signal(s, forward) for s in _FORWARDED}
        try:
            code = child.wait()
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)

    stopped_by = signal.Signals(received[0]).name if received else None
    if code < 0:  # the child died of a signal itself (e.g. the OOM killer's SIGKILL)
        stopped_by = stopped_by or signal.Signals(-code).name
        code = 128 - code
    return _finish(run_dir, started, code, stopped_by)


def _finish(run_dir: Path, started: RunStatus, code: int, stopped_by: str | None = None) -> int:
    """Record that the run ended (when, its exit code, the signal) and return the code."""
    write_status(
        run_dir,
        RunStatus(
            started_at=started.started_at,
            host=started.host,
            finished_at=time.time(),
            exit_code=code,
            signal=stopped_by,
        ),
    )
    return code


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m almasim.jobs.wrapper <run dir>", file=sys.stderr)
        return 2
    return run_job(args[0], job_command())


if __name__ == "__main__":
    sys.exit(main())
