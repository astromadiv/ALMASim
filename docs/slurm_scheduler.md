# Slurm Scheduler

This page describes the Slurm-backed Dask cluster helper in
[`src/almasim/scheduling/cluster.py`](../src/almasim/scheduling/cluster.py).
The `slurm` compute backend (`SlurmBackend`, see [Compute Backends](compute.md))
is built on it, and so is every command that takes `--backend slurm`:
`products download`, `visibilities extract`/`unpack`/`calibrate`,
`image image-from-ms`, `predict` and `noise extract`/`fit`/`validate`.

To run a whole `almasim` command as one Slurm batch job instead, use
`almasim submit` or the [web console](console.md); see
[`almasim submit`](cli.md#almasim-submit).

## Overview

The module provides:

- `SlurmDaskClusterSingleton`: a shared wrapper around
  `dask_jobqueue.slurm.SLURMCluster` and `dask.distributed.Client`
- `run_subcommand(...)`: runs a command on a worker and returns a
  `SubcommandResult` with `stdout`, `stderr` and `returncode`
- `default_scheduler_host(queue)`: the address the workers of a partition can
  reach the scheduler on

## Main behavior

### Cluster singleton

`SlurmDaskClusterSingleton.get_instance(...)` creates one Slurm-backed Dask
cluster per process and returns it to every later caller with the same
settings. Each caller holds the cluster until it calls `release_instance()`;
the last release closes it, and `close_instance()` closes it at once whoever
still holds it. A call with different settings while the cluster is running
raises `SlurmClusterBusyError`.

Each worker is one Slurm job (`n_jobs` of them) with `node_cores` cores,
`memory` and `walltime`, on partition `queue` and account `project`.
`SlurmBackend` also adds `--requeue` to the job directives unless you pass
`--requeue` or `--no-requeue` yourself.

### CPU resource model

- Each worker declares the resource `CPU=<node_cores>`.
- Each submitted task (`submit_subcommand` or `submit_callable`) requests
  `resources={"CPU": cores}`.
- `cores` must be less than `node_cores`; `run_subcommand` also checks it
  against the worker's own CPU count.

So a worker runs as many tasks at once as fit in its cores, and each task
stays below the cores of a node. In `image image-from-ms` and `predict`,
`--num-cores` is the task's `cores` and `--max-cores-per-node` the worker's
`node_cores`.

`run_subcommand` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to the
task's `cores`, and puts the system library directories first on
`LD_LIBRARY_PATH`. A command that exceeds its `timeout` returns code 124.

### Scheduler host and interface

The workers must reach the scheduler, which runs where the command was
started. On systems with several networks you can set how it advertises
itself:

- `scheduler_host` (`--scheduler-host`): the host or IP address the scheduler
  advertises to the workers
- `scheduler_interface` (`--scheduler-interface`): the network interface
  (e.g. `ib0`) the scheduler and workers use

Give one or the other: Dask refuses a host together with an interface. With
neither, `default_scheduler_host(queue)` picks the local address the kernel
routes to the partition's nodes (found with `sinfo`), and falls back to the
submit host's `HOSTNAME` unless that resolves to loopback.

### Dashboard

The scheduler serves the Dask dashboard (with bokeh installed: the
`dashboard` extra) on a free port, logged as `Dask dashboard: <link>`; pass
`scheduler_options={"dashboard_address": ":8787"}` for a fixed one. Inside a
console job (the job wrapper sets `$ALMASIM_RUN_DIR`) it also writes the
address to `dashboard.json` in the run folder, and removes it when the
cluster closes; the console relays the dashboard from there
([console](console.md#jobs-and-a-jobs-page)).

### Worker logs

The workers' stdout and stderr go to `~/dask-worker-logs` (the
`log_directory` option, which must stay inside your home directory). When a
worker log shows that it could not connect to the scheduler,
`SlurmBackend.wait_for_workers` stops waiting and raises an error that names
the log, instead of waiting for the full timeout.

## API sketch

```python
from almasim.scheduling.cluster import SlurmDaskClusterSingleton

manager = SlurmDaskClusterSingleton.get_instance(
    queue="normal",
    node_cores=8,
    memory="16GB",
    walltime="00:30:00",
    n_jobs=1,
    project=None,
    scheduler_interface="ib0",  # optional; or scheduler_host="headnode-internal"
)

future = manager.submit_subcommand(
    command=["bash", "-lc", "echo hello"],
    cores=2,
    timeout=30,
)
result = future.result(timeout=120)

print(result.returncode)
print(result.stdout)
print(result.stderr)

SlurmDaskClusterSingleton.release_instance()
```

## Demo script

A complete runnable example is in
[`examples/slurm_cluster_submit_demo.py`](../examples/slurm_cluster_submit_demo.py).

### What the demo does

1. Creates the singleton cluster
2. Waits for at least one worker
3. Submits a shell command with CPU constraints
4. Prints the command result (`stdout`, `stderr`, `returncode`)
5. Closes the singleton

On a timeout or failure it prints `squeue`, the recent Dask jobs from `sacct`
and the `scontrol` details of the last one.

### Useful options

- `--command`: the command to run on the worker
- `--cores`: cores the task requests (less than `--node-cores`)
- `--queue`: Slurm partition
- `--project`: Slurm account
- `--node-cores`: total cores per node
- `--memory`, `--walltime`: memory and walltime per worker job
- `--n-jobs`: number of Slurm jobs (workers)
- `--scheduler-host`: explicit scheduler host advertised to workers
- `--scheduler-interface`: network interface for communication
- `--worker-start-timeout`: how long to wait for the first worker
- `--result-timeout`: how long to wait for the task's result
- `--task-timeout`: timeout of the worker-side subprocess

### Example command

```bash
python examples/slurm_cluster_submit_demo.py \
  --command "echo hello" \
  --cores 2 \
  --node-cores 8 \
  --n-jobs 1 \
  --scheduler-interface ib0 \
  --worker-start-timeout 60 \
  --result-timeout 120 \
  --task-timeout 30
```

## Troubleshooting

If workers do not appear, check:

1. That the scheduler address the demo prints is reachable from the compute
   nodes.
2. The worker logs in `~/dask-worker-logs` (`dask-worker-<jobid>.err`).
3. `sacct` for failed `dask-worker` jobs.
4. `scontrol show job <jobid>` for the job's `StdOut`/`StdErr` paths.
5. Partition and account limits, and firewall rules.

A common failure is a worker timing out while connecting to a scheduler on an
unreachable host or interface. The fix is to start the command from a compute
node (for example inside an `sbatch` job, or with `almasim submit`) or to pass
`--scheduler-host` or `--scheduler-interface` with an address the nodes can
reach.
