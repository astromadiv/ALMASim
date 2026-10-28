# ALMASim for Open OnDemand

An Open OnDemand interactive app that runs the ALMASim web console as a
small Slurm job of the user's, on a compute node, reached through the
portal's proxy at `/node/<host>/<port>/`. The console itself is described in
[docs/console.md](../../docs/console.md); this app starts it with
`almasim serve --managed` instead of the user running `almasim serve` on the
login node.

| File | What it does |
| --- | --- |
| `manifest.yml` | The app's name and place in the portal's menu. |
| `form.yml` | The launch form: account, partition, hours, memory; the cluster's name and where `almasim` is installed. |
| `submit.yml.erb` | The job: one CPU and the memory asked for. |
| `template/before.sh.erb` | Picks a free port and a fresh token, and the path the proxy forwards. |
| `template/script.sh.erb` | Starts `almasim serve --managed` with them. |
| `view.html.erb` | The Connect button: opens the console with the token once. |
| `try-without-ondemand.sh` | Not part of the app: rehearses it without a portal (below). |

## Trying it without a portal

`try-without-ondemand.sh` starts the console as a Slurm job the way the
portal would, running this app's own `before.sh` and `script.sh`, then
prints an SSH tunnel (standing in for the portal's proxy) and the link the
Connect button would open:

```bash
bash ood/almasim/try-without-ondemand.sh                 # 1 CPU, 4 GB, 8 hours
bash ood/almasim/try-without-ondemand.sh -p PART -A ACC  # other sbatch options
ALMASIM_SETUP=~/dev-env.sh bash ood/almasim/try-without-ondemand.sh
```

`ALMASIM_SETUP` is the script that puts `almasim` on the PATH; by default the
one `form.yml` names. The install must have `almasim serve --managed`, or
the session stops and says so. Each session keeps its files (the job's log
among them) in `~/ondemand-try/<session>/`. Stop a session with
`scancel <job id>`.

## Installing

1. Set `cluster` and `almasim_setup` in `form.yml` for the site.
2. Copy this folder to `/var/www/ood/apps/sys/almasim/` (a system app), or
   to `~/ondemand/dev/almasim/` to try it as one user first.
3. The portal's proxy forwards to `/node/<host>/<port>/`, so `<host>` must be a
   name the portal host resolves, and `host_regex` must match it. On zwicky
   the long names `hostname` gives (`compute-node47.zwicky.…`) do not even
   resolve on the login node; the short ones do. So set the cluster's
   `batch_connect.basic.set_host` to `host=$(hostname -s)`, and `host_regex`
   to match `compute-node\d+`.

Nothing else is needed: the console runs as the user, with the user's
Slurm account and files, and stops when the session ends.
