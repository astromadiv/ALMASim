#!/bin/bash
# The console the end-to-end tests drive (web console plan, U3.9): `almasim
# serve` on a port, in a throwaway area (runs, a data root, a home), every
# command on the console's own machine (executor = "server"), and each
# command replaced by e2e/fake_almasim.py, which finishes in seconds.
#
#   E2E_ROOT   the area (default /tmp/almasim-e2e; emptied first)
#   E2E_PORT   the port (default 8765)
#   E2E_BASE_PATH  a URL prefix to serve under, as Open OnDemand does (default none)
#   ALMASIM    the almasim to run (default: almasim on PATH)
#   PYTHON     the Python with numpy and astropy for the fake (default: python)
set -euo pipefail

root=${E2E_ROOT:-/tmp/almasim-e2e}
here=$(cd "$(dirname "$0")" && pwd)
rm -rf "$root"
mkdir -p "$root/home" "$root/runs" "$root/data/cal" "$root/data/images" "$root/noise-models"
cat > "$root/profile.toml" <<TOML
executor = "server"
runs_root = "$root/runs"
roots = ["$root/data"]

[tools]
noise_models = "$root/noise-models"
TOML

export HOME="$root/home"
export ALMASIM_SITE_PROFILE="$root/profile.toml"
export ALMASIM_TOKEN=${ALMASIM_TOKEN:?the tests set the token}
export ALMASIM_WEB_DIR="$here/../build"
export ALMASIM_JOB_COMMAND="${PYTHON:-python} $here/fake_almasim.py"
export ALMASIM_BASE_PATH="${E2E_BASE_PATH:-}"
exec "${ALMASIM:-almasim}" serve --port "${E2E_PORT:-8765}"
