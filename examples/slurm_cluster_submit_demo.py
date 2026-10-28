"""Small demo for submitting shell subcommands through the Slurm Dask singleton.

Example:
    python examples/slurm_cluster_submit_demo.py --command "echo hello" --cores 2
"""

from __future__ import annotations

import argparse
import re
import subprocess
from concurrent.futures import TimeoutError as FutureTimeoutError

from almasim.scheduling.cluster import SlurmDaskClusterSingleton


def _run_shell(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (completed.stdout or completed.stderr or "").strip()


def _print_slurm_debug_info() -> None:
    print("=== SLURM Debug Info ===")

    squeue_out = _run_shell(
        [
            "squeue",
            "-u",
            "$(whoami)",
            "-o",
            "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R",
        ]
    )
    # squeue does not expand shell variables in direct exec; retry with USER env fallback.
    if "Invalid user" in squeue_out or not squeue_out:
        user = _run_shell(["whoami"])
        squeue_out = _run_shell(
            [
                "squeue",
                "-u",
                user,
                "-o",
                "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R",
            ]
        )
    print("squeue:\n" + squeue_out)

    sacct_out = _run_shell(
        [
            "sacct",
            "-S",
            "now-30minutes",
            "-u",
            _run_shell(["whoami"]),
            "--format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList,Reason",
            "-P",
        ]
    )
    dask_lines = [line for line in sacct_out.splitlines() if "dask" in line.lower()]
    if dask_lines:
        print("recent dask jobs from sacct:")
        print("\n".join(dask_lines[-10:]))

        last_job_id = dask_lines[-1].split("|", maxsplit=1)[0].strip().split(".")[0]
        scontrol_out = _run_shell(["scontrol", "show", "job", last_job_id])
        print(f"scontrol job {last_job_id}:")
        print(scontrol_out)

        stdout_match = re.search(r"\bStdOut=(\S+)", scontrol_out)
        stderr_match = re.search(r"\bStdErr=(\S+)", scontrol_out)
        if stdout_match:
            print(f"job stdout path: {stdout_match.group(1)}")
        if stderr_match:
            print(f"job stderr path: {stderr_match.group(1)}")
    else:
        print("No recent dask jobs found in sacct output.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SLURM Dask singleton submit demo")
    parser.add_argument(
        "--command",
        default="echo hello-from-worker",
        help="Command to run on the worker",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Requested task cores; must be less than node cores",
    )
    parser.add_argument("--queue", default="normal", help="SLURM queue/partition")
    parser.add_argument("--node-cores", type=int, default=8, help="Cores available per node")
    parser.add_argument("--memory", default="16GB", help="Memory per worker/job")
    parser.add_argument("--walltime", default="00:30:00", help="Walltime HH:MM:SS")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of SLURM jobs/workers")
    parser.add_argument("--project", default=None, help="Optional SLURM project/account")
    parser.add_argument(
        "--scheduler-host",
        default=None,
        help=(
            "Host advertised by scheduler (defaults to the address routed to the "
            "partition's nodes; give this or --scheduler-interface, not both)"
        ),
    )
    parser.add_argument(
        "--scheduler-interface",
        default=None,
        help="Network interface for scheduler/worker communication (e.g. ib0, eth0)",
    )
    parser.add_argument(
        "--worker-start-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for at least one worker before failing",
    )
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for task result before failing",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=60.0,
        help="Seconds for the worker-side subprocess timeout",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    manager = SlurmDaskClusterSingleton.get_instance(
        queue=args.queue,
        node_cores=args.node_cores,
        memory=args.memory,
        walltime=args.walltime,
        n_jobs=args.n_jobs,
        project=args.project,
        scheduler_host=args.scheduler_host,
        scheduler_interface=args.scheduler_interface,
    )

    try:
        print(f"Scheduler address: {manager.cluster.scheduler_address}")
        print("Waiting for at least one SLURM worker to start...")
        manager.client.wait_for_workers(1, timeout=args.worker_start_timeout)

        print("Submitting subcommand...")
        future = manager.submit_subcommand(
            command=args.command,
            cores=args.cores,
            timeout=args.task_timeout,
        )
        result = future.result(timeout=args.result_timeout)
    except (FutureTimeoutError, TimeoutError):
        print(
            "Timed out waiting for result. "
            "Increase --result-timeout or check SLURM queue/worker startup."
        )
        _print_slurm_debug_info()
        return 124
    except Exception as exc:  # noqa: BLE001 - a demo reports any failure
        print(f"Cluster startup/submit failed: {exc}")
        _print_slurm_debug_info()
        return 1
    finally:
        SlurmDaskClusterSingleton.close_instance()

    print("=== Subcommand Result ===")
    print(f"returncode: {result.returncode}")
    print("stdout:")
    print(result.stdout.rstrip())
    print("stderr:")
    print(result.stderr.rstrip())
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
