#!/bin/bash
# A shared ALMASim install for a team on a cluster (web console plan, M4.2).
#
#   scripts/install_shared.sh <prefix> <site profile template> [wheel]
#
# makes, under <prefix> (e.g. /SKAO_data/cycle11/almasim):
#   python/       the Python 3.12 the environment runs (uv's own build), here
#                 and not in the installer's home, which others cannot enter
#   env/          a Python 3.12 environment with almasim, its web console,
#                 casatools/casatasks, python-casacore and bokeh (for the
#                 Dask dashboard): the Python every job runs
#   site.toml     the site profile: the template, with that Python filled in
#                 (and its noise_models line, if any, pointing at noise-models/)
#   noise-models/ the team's own noise models, group-writable: the console
#                 releases them here (when the template has a noise_models line)
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

# Python itself under the prefix: uv otherwise keeps its Pythons in the
# installer's home, and env/bin/python would link there, out of the group's
# reach ("bad interpreter: Permission denied" for everyone else).
export UV_PYTHON_INSTALL_DIR="$prefix/python"
uv python install --no-bin 3.12  # not into the installer's own ~/.local/bin
uv venv --allow-existing --managed-python --python 3.12 "$prefix/env"
real_prefix=$(readlink -f "$prefix")
case $(readlink -f "$prefix/env/bin/python") in
  "$real_prefix"/python/*) ;;
  *) echo "$prefix/env/bin/python is not the Python in $prefix/python" >&2; exit 1 ;;
esac
# The versions tested (uv.lock), not the newest: a newer casatools, say,
# can change what calibration gives.
uv export --project "$repo" --frozen --no-hashes --no-emit-project --extra web --extra casa \
  --extra ms-casacore --extra dashboard --extra mcp --no-default-groups > "$prefix/constraints.txt"
uv pip install --python "$prefix/env/bin/python" --constraints "$prefix/constraints.txt" \
  --reinstall-package almasim "$wheel[web,casa,ms-casacore,dashboard,mcp]"

python_line="executable = \"$prefix/env/bin/python\""
noise_line="noise_models = \"$prefix/noise-models\""
sed -e "s|^executable = .*|$python_line|" -e "s|^noise_models = .*|$noise_line|" \
  "$template" > "$prefix/site.toml"
grep -q "^$python_line" "$prefix/site.toml" || { echo "$template has no [python] executable line" >&2; exit 1; }
# Released into by anyone in the group (setgid: new files stay the group's);
# made once, never changed afterwards: its files belong to whoever released them.
if grep -q "^$noise_line" "$prefix/site.toml" && [ ! -d "$prefix/noise-models" ]; then
  install -d -m 2775 "$prefix/noise-models"
fi

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
