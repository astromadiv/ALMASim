#!/bin/bash
# A shared ALMASim install for a team on a cluster (web console plan, M4.2).
#
#   scripts/install_shared.sh <prefix> <site profile template> [wheel]
#
# makes, under <prefix> (e.g. /SKAO_data/cycle11/almasim):
#   env/          a Python 3.12 environment with almasim, its web console,
#                 casatools/casatasks and python-casacore: the Python every
#                 job runs
#   site.toml     the site profile: the template, with that Python filled in
#   almasim.sh    what users source: puts almasim on PATH, names the profile
#   install.env   where it came from (repo, commit, template), which
#                 `almasim update-shared` builds the next version from
#
# The wheel (default: built here from this checkout, the web app included)
# needs `npm run build` in frontend/ first. Run it again with a new wheel to
# update: the environment is reinstalled in place, the profile rewritten.
# Or let `almasim update-shared` do all of it from a git ref, as a Slurm job.
# Needs uv, and internet to fetch Python and the packages (on zwicky: run it
# on a compute node).
set -euo pipefail

prefix=${1:?usage: install_shared.sh <prefix> <site profile template> [wheel]}
template=${2:?usage: install_shared.sh <prefix> <site profile template> [wheel]}
wheel=${3:-}
given_wheel=$wheel
repo=$(cd "$(dirname "$0")/.." && pwd)

umask 002  # the group shares the install
mkdir -p "$prefix"

if [ -z "$wheel" ]; then
  [ -f "$repo/frontend/build/200.html" ] || { echo "build the web app first: cd frontend && npm ci && npm run build" >&2; exit 1; }
  rm -rf "$prefix/dist"
  ALMASIM_REQUIRE_WEB=1 uv build --wheel --out-dir "$prefix/dist" "$repo"
  wheel=$(ls "$prefix"/dist/almasim-*.whl)
fi

uv venv --allow-existing --python 3.12 "$prefix/env"
# The versions tested (uv.lock), not the newest: a newer casatools, say,
# can change what calibration gives.
uv export --project "$repo" --frozen --no-hashes --no-emit-project --extra web --extra casa \
  --extra ms-casacore --no-default-groups > "$prefix/constraints.txt"
uv pip install --python "$prefix/env/bin/python" --constraints "$prefix/constraints.txt" \
  --reinstall-package almasim "$wheel[web,casa,ms-casacore]"

python_line="executable = \"$prefix/env/bin/python\""
sed "s|^executable = .*|$python_line|" "$template" > "$prefix/site.toml"
grep -q "^$python_line" "$prefix/site.toml" || { echo "$template has no [python] executable line" >&2; exit 1; }

cat > "$prefix/almasim.sh" <<SH
# ALMASim, shared install: source this file (e.g. from ~/.bashrc).
export PATH="$prefix/env/bin:\$PATH"
export ALMASIM_SITE_PROFILE="$prefix/site.toml"
SH

# The template relative to the repo, so the next update finds it in its clone.
template_path=$(cd "$(dirname "$template")" && pwd)/$(basename "$template")
case $template_path in
  "$repo"/*) template_path=${template_path#"$repo"/} ;;
esac
commit=
[ -n "$given_wheel" ] || commit=$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)
cat > "$prefix/install.env" <<ENV
repo=$(git -C "$repo" remote get-url origin 2>/dev/null || true)
commit=$commit
template=$template_path
ENV

version=$("$prefix/env/bin/python" -c "from almasim.__version__ import __version__; print(__version__)")
echo "installed almasim $version in $prefix"
echo "users: source $prefix/almasim.sh, then: almasim doctor --slurm; almasim serve"
