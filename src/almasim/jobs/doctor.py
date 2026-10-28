"""Check that a profile can run jobs here (web console plan, J1.3).

Each check runs where ``almasim doctor`` runs. Run it on a compute node
(``almasim doctor --slurm``) to see what a job will see: the login node can
have tools the compute nodes lack, and the other way round.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

from almasim.services.runtime.env import casa_environment, wsclean_environment

from .profile import Profile
from .rundir import is_group_shared


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    # A missing optional tool is reported but does not fail the doctor.
    required: bool = True


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run(cmd: list[str], env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False, env=env
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "timed out after 120 s")


# Checked the way the pipeline starts CASA children: casatools' bundled
# libraries first on LD_LIBRARY_PATH (runtime.env.casa_environment; without
# them the import hung on zwicky) and a site config with automatic data
# updates off (casa.environment): concurrent jobs must not update a shared
# CASA data folder, and some sites' compute nodes have no internet.
_CASA_LIB_DIR = (
    "import importlib.util, os\n"
    "spec = importlib.util.find_spec('casatools')\n"
    "pkg = os.path.dirname(spec.origin) if spec and spec.origin else ''\n"
    "lib = os.path.join(pkg, '__casac__', 'lib')\n"
    "print(lib if pkg and os.path.isdir(lib) else '')\n"
)
_CASA_IMPORT = (
    "import os, sys, tempfile\n"
    "config = os.path.join(tempfile.mkdtemp(), 'casasiteconfig.py')\n"
    "with open(config, 'w') as f:\n"
    "    f.write(f'measurespath = {sys.argv[1]!r}\\n'\n"
    "            'data_auto_update = False\\nmeasures_auto_update = False\\n')\n"
    "os.environ['CASASITECONFIG'] = config\n"
    "import casatools, casatasks\n"
    "print(casatools.version_string() if hasattr(casatools, 'version_string') else 'imported')\n"
)
# python-casacore reads MeasurementSet metadata (image compute-parameters and
# the noise tools); it is an optional extra, so an install can lack it.
_CASACORE_IMPORT = "import casacore, casacore.tables; print(casacore.__version__)"
# A Dask scheduler serves its dashboard (which the console relays) only with
# bokeh, the dashboard extra; without it jobs run the same, just unwatched.
_BOKEH_IMPORT = "import bokeh; print(bokeh.__version__)"


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _runs_root(profile: Profile) -> Check:
    root = profile.runs_root
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, prefix=".doctor-"):
            pass
    except OSError as exc:
        return Check("runs root", False, f"{root}: {exc.strerror or exc}")
    shared = "group-shared (setgid)" if is_group_shared(root) else "private (no setgid bit)"
    return Check("runs root", True, f"{root}, writable, {shared}")


def _python(profile: Profile, run: Runner) -> Iterator[Check]:
    python = profile.python
    if not (os.path.isfile(python) and os.access(python, os.X_OK)):
        yield Check("python", False, f"{python} is not an executable file")
        return
    result = run(
        [python, "-c", "import almasim, sys; print(almasim.__file__, sys.version.split()[0])"]
    )
    if result.returncode != 0:
        yield Check("python", False, f"{python} cannot import almasim: {_last_line(result.stderr)}")
        return
    yield Check("python", True, f"{python} (Python {result.stdout.split()[-1]})")
    measures = str(profile.casa_data or tempfile.gettempdir())
    lib_dir = run([python, "-c", _CASA_LIB_DIR]).stdout.strip() or None
    casa = run([python, "-c", _CASA_IMPORT, measures], env=casa_environment(bundled=lib_dir))
    yield Check(
        "casatools",
        casa.returncode == 0,
        f"casatools {casa.stdout.strip()}"
        if casa.returncode == 0
        else f"not importable in {python}: {_last_line(casa.stderr)}",
        required=False,
    )
    casacore = run([python, "-c", _CASACORE_IMPORT])
    yield Check(
        "casacore",
        casacore.returncode == 0,
        f"python-casacore {casacore.stdout.strip()}"
        if casacore.returncode == 0
        else f"not importable in {python} (the ms-casacore extra): {_last_line(casacore.stderr)}",
        required=False,
    )
    bokeh = run([python, "-c", _BOKEH_IMPORT])
    yield Check(
        "dashboard",
        bokeh.returncode == 0,
        f"bokeh {bokeh.stdout.strip()}: jobs' Dask dashboards"
        if bokeh.returncode == 0
        else f"no bokeh in {python} (the dashboard extra), so jobs serve no Dask dashboard",
        required=False,
    )


def _wsclean(profile: Profile, run: Runner) -> Check:
    binary = profile.wsclean or shutil.which("wsclean")
    if not binary:
        return Check("wsclean", False, "not set in [tools] and not on PATH", required=False)
    try:
        # WSClean refuses to start with multi-threaded OpenBLAS (wsclean_environment).
        result = run([binary, "--version"], env=wsclean_environment(1))
    except OSError as exc:
        return Check("wsclean", False, f"{binary}: {exc.strerror or exc}", required=False)
    version = next((line for line in result.stdout.splitlines() if "WSClean version" in line), "")
    return Check("wsclean", result.returncode == 0, version.strip() or binary, required=False)


def _casa_data(profile: Profile) -> Check:
    if profile.casa_data is None:
        return Check("casa data", False, "not set in [tools]", required=False)
    ok = profile.casa_data.is_dir()
    return Check("casa data", ok, f"{profile.casa_data}{'' if ok else ' does not exist'}", False)


def _roots(profile: Profile) -> Check:
    missing = [str(p) for p in profile.roots if not p.expanduser().is_dir()]
    if missing:
        return Check("roots", False, f"missing: {', '.join(missing)}")
    return Check("roots", True, ", ".join(str(p) for p in profile.roots))


def _slurm(profile: Profile, run: Runner) -> Iterator[Check]:
    missing = [tool for tool in ("sbatch", "squeue", "sacct", "scancel") if not shutil.which(tool)]
    if missing:
        yield Check("slurm", False, f"not on PATH: {', '.join(missing)}")
        return
    result = run(["sinfo", "-h", "-o", "%P"])
    available = {p.rstrip("*") for p in result.stdout.split()}
    yield Check("slurm", result.returncode == 0, f"partitions: {', '.join(sorted(available))}")
    wanted: set[str] = set(profile.partitions)
    if profile.partition:
        wanted.add(profile.partition)
    wanted.update(p.partition for p in profile.commands.values() if p.partition)
    unknown = sorted(wanted - available)
    if unknown:
        yield Check("partitions", False, f"not on this cluster: {', '.join(unknown)}")


def run_checks(profile: Profile, run: Runner = _run) -> list[Check]:
    """Every check, in the order ``almasim doctor`` prints them."""
    sources = ", ".join(str(s) for s in profile.sources) or "none (built-in defaults)"
    checks = [Check("profile", True, sources), _runs_root(profile), _roots(profile)]
    checks.extend(_python(profile, run))
    checks.append(_wsclean(profile, run))
    checks.append(_casa_data(profile))
    checks.extend(_slurm(profile, run))
    return checks


def srun_command(profile: Profile, argv: list[str]) -> list[str]:
    """``argv`` (an almasim command line) run on a compute node with ``srun``."""
    cmd = ["srun", "--cpus-per-task=2", "--mem=4G", "--time=00:10:00"]
    if profile.partition:
        cmd.append(f"--partition={profile.partition}")
    if profile.account:
        cmd.append(f"--account={profile.account}")
    return [*cmd, profile.python, "-m", "almasim.cli", *argv]


def format_report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = []
    for c in checks:
        mark = "ok  " if c.ok else ("FAIL" if c.required else "warn")
        lines.append(f"{mark}  {c.name:<{width}}  {c.detail}")
    return "\n".join(lines)


def passed(checks: list[Check]) -> bool:
    return all(c.ok or not c.required for c in checks)


__all__ = ["Check", "format_report", "passed", "run_checks", "srun_command"]
