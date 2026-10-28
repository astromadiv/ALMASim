# Compute Backends

This page describes ALMASim's compute backends.

## Location

Compute backends live in [`src/almasim/services/compute`](../src/almasim/services/compute).

## Supported backends

| backend | class | runs tasks | use for |
|---|---|---|---|
| `sync` | `SyncBackend` | one after another in the current process | notebooks, examples, debugging |
| `local` | `LocalBackend` | in a pool of worker processes on this machine (`n_workers`, default: the CPU count) | parallelism on one node |
| `dask` | `DaskBackend` | on a Dask cluster: an existing scheduler (`scheduler="tcp://host:8786"`) or a local process cluster | distributed execution on a workstation or cluster |
| `slurm` | `SlurmBackend` | on Dask workers started as Slurm jobs (`dask-jobqueue`) | HPC clusters; see [Slurm Scheduler](slurm_scheduler.md) |
| `kubernetes` | `KubernetesBackend` | on Dask workers in Kubernetes pods (`dask-kubernetes`) | cluster-native environments |

## Main modules

- `base.py`: `ComputationBackend`, the common interface (`scatter`,
  `compute`, `gather`, `delayed`, `close`; every backend is a context
  manager), and `DaskClientBackend`, shared by the Dask-based backends
- `sync.py`, `local.py`, `dask.py`, `slurm.py`, `kubernetes.py`: the backends
- `factory.py`: `create_backend(backend_type, **kwargs)`
- `progress.py`: progress, heartbeat and failure reporting shared by the
  archive stages, the imaging runs and the simulator

```python
from almasim.services.compute import create_backend
from almasim.services.simulation.stages import generate_clean_cube, simulate_observation

# params: a SimulationParams, e.g. SimulationParams.from_metadata_row(row, ...)
with create_backend("local", n_workers=8) as backend:
    clean = generate_clean_cube(params, compute_backend=backend)
    results = simulate_observation(clean, compute_backend=backend)
```

## Where each backend is used

- **Simulations**: `almasim simulation --backend` takes `sync` (default) or
  `local` with `--n-workers`. The Python stages take any backend through
  `compute_backend=`; without one they use `sync`.
- **Archive, imaging and noise commands**: `products download`,
  `visibilities extract`/`unpack`/`calibrate`, `image image-from-ms` and
  `noise extract`/`fit`/`validate` take `--backend sync` or `--backend slurm`
  with the `--slurm-*` options; `predict` takes `--use-slurm`. On `slurm`
  each task runs on a Dask worker started as a Slurm job.
- **Whole commands as Slurm jobs**: `almasim submit` and the
  [web console](console.md) run an entire `almasim` command as one Slurm batch
  job. That is the job layer in [`src/almasim/jobs`](../src/almasim/jobs), not
  a compute backend; a submitted command can still use any backend inside its
  job.

## Design goal

The simulation service layer does not need to know which backend is active.
The caller chooses the backend and passes it to the workflow, and the
simulation code uses only the common interface.

## Slurm details

For Slurm scheduler usage and troubleshooting, see
[Slurm Scheduler](slurm_scheduler.md).
