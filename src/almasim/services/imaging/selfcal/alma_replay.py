"""The ALMA pipeline's self-calibration of a member, replayed from its CASA commands.

A delivery holds the calibration tables of the execution blocks (``s<n>.caltables.tgz``)
but not those of the imaging pipeline's self-calibration (``hif_selfcal``): ALMA's
restore script rebuilds the calibrated data without them. The member's
``log/*.casa_commands.log`` lists every CASA call the pipeline made, with its
parameters as they were computed (thresholds, solution intervals, reference
antennas), and which rounds it kept shows in the calls themselves: a round it
rejected is followed by the restore of the flags and no ``applycal``.
:func:`replay_member` runs those calls again on the member's calibrated
MeasurementSets and keeps the gain tables the pipeline applied to its targets,
with how it applied them (``apply.json``). :func:`apply_selfcal` applies them to
one window of one execution block, as ``image-from-ms --alma-selfcal`` does
before regridding the window.

The replay, as the pipeline did it:

1. ``hif_mstransform``: each execution block's targets, its science windows at
   full resolution (``<eb>_targets.ms``), from the calibrated MS's DATA (the
   pipeline's CORRECTED_DATA).
2. ``hifa_flagtargets``: the commands of the target flagging template, which the
   weblog keeps (``<eb>.flagtargetstemplate.txt``).
3. ``hif_selfcal``: the line channels flagged, a channel-averaged copy of each
   ``<eb>_targets.ms`` in a working folder of its own, and on it every round's
   ``tclean``, ``gaincal`` and ``applycal``. The calls that apply the final tables to
   the full-resolution data (``applycal`` with a calibration library, which the
   delivery does not hold) end the replay: the tables and the last ``applycal`` of
   each execution block and field are what they applied.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import shutil
import tarfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

APPLY_FILE = "apply.json"
"""The file of a replay's folder that says which tables to apply, and how."""

FOLDER = "alma-selfcal"
"""A replay's folder, in a member's ``images/`` folder or an output's member folder."""

_CALL_START = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(")
_STAGE = re.compile(r"^# (h[a-z]*_[a-z_]+)\(")
_TARGETS_SUFFIX = "_targets.ms"


@dataclass(frozen=True)
class CasaCall:
    """One call of a CASA commands log: the task, its arguments, its line."""

    task: str
    kwargs: dict[str, Any]
    line: int
    args: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Stage:
    """A pipeline stage of the log (``hif_selfcal``) and its calls in order."""

    name: str
    calls: tuple[CasaCall, ...]


def _join(text: str, line: str) -> str:
    """The log wraps calls at 80 columns, inside strings too: at a space, which the
    wrap drops, or after a word's hyphen (``'auto-`` / ``multithresh'``)."""
    line = line.strip()
    if re.search(r"[A-Za-z]-$", text) and re.match(r"[A-Za-z]", line):
        return text + line
    return f"{text} {line}" if text else line


class _NumpyScalars(ast.NodeTransformer):
    """``np.str_('G')`` → ``'G'``: the pipeline logs a NumPy scalar argument by its repr."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in ("np", "numpy")
            and len(node.args) == 1
            and not node.keywords
        ):
            return self.visit(node.args[0])
        return self.generic_visit(node)


def _literal(node: ast.expr) -> Any:
    return ast.literal_eval(_NumpyScalars().visit(node))


def _parse_call(text: str, line: int) -> CasaCall:
    node = ast.parse(text, mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"line {line}: not a call: {text[:80]}")
    kwargs = {k.arg: _literal(k.value) for k in node.keywords if k.arg}
    args = tuple(_literal(a) for a in node.args)
    return CasaCall(task=node.func.id, kwargs=kwargs, line=line, args=args)


def parse_casa_commands(text: str) -> list[Stage]:
    """The stages of a ``casa_commands.log`` and their calls."""
    stages: list[tuple[str, list[CasaCall]]] = [("", [])]
    buffer, start = "", 0
    for number, line in enumerate(text.splitlines(), 1):
        if not buffer:
            stage = _STAGE.match(line)
            if stage:
                stages.append((stage.group(1), []))
                continue
            if not _CALL_START.match(line):
                continue
            start = number
        buffer = _join(buffer, line)
        try:
            call = _parse_call(buffer, start)
        except SyntaxError:
            continue  # the call goes on on the next line
        stages[-1][1].append(call)
        buffer = ""
    if buffer:
        raise ValueError(f"line {start}: a call that does not end: {buffer[:80]}")
    return [Stage(name, tuple(calls)) for name, calls in stages if name or calls]


def stage_calls(stages: Iterable[Stage], name: str) -> list[CasaCall]:
    """Every call of the stages called ``name``, in order."""
    return [call for stage in stages if stage.name == name for call in stage.calls]


# -- the plan: which call runs where ------------------------------------------


@dataclass(frozen=True)
class Action:
    """One CASA (or file) call of the replay, its paths made absolute."""

    task: str
    kwargs: dict[str, Any]
    line: int
    cwd: str | None = None


@dataclass
class ReplayPlan:
    """The replay of one member, and what it keeps."""

    actions: list[Action] = field(default_factory=list)
    # The last applycal of each (execution block, field) before the final one.
    applied: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    # The execution blocks the pipeline applied its tables to (callib applycal).
    finally_applied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def eb_of(vis: str) -> str:
    """``uid___A002_X1_X2_targets.ms`` (or ``.ms``) -> ``uid___A002_X1_X2``."""
    name = Path(str(vis)).name
    return name.removesuffix(_TARGETS_SUFFIX).removesuffix(".ms")


def _flag_commands(template: str | None) -> list[str]:
    """The commands of a flagging template: its lines that are not comments."""
    if not template:
        return []
    return [
        line.strip()
        for line in Path(template).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def plan_replay(
    stages: Sequence[Stage],
    calibrated: Mapping[str, str],
    workdir: str | os.PathLike[str],
    flag_templates: Mapping[str, str] | None = None,
) -> ReplayPlan:
    """The replay of ``stages`` on the ``calibrated`` MSs (``{eb: path}``) in ``workdir``.

    ``workdir/targets`` takes each EB's ``<eb>_targets.ms``; ``workdir/selfcal``
    is the self-calibration's own folder, which its averaged copies, images and
    tables are in (its relative paths are relative to it).
    """
    work = Path(workdir)
    plan = ReplayPlan()
    _plan_targets(plan, stages, calibrated, work / "targets")
    _plan_target_flags(plan, stages, flag_templates or {}, work / "targets")
    _SelfcalRouting(work / "targets", work / "selfcal").plan(plan, stages)
    return plan


def _plan_targets(
    plan: ReplayPlan, stages: Sequence[Stage], calibrated: Mapping[str, str], targets: Path
) -> None:
    """``hif_mstransform``: each EB's targets from its calibrated MS."""
    for call in stage_calls(stages, "hif_mstransform"):
        if call.task != "mstransform":
            continue
        eb = eb_of(call.kwargs["vis"])
        if eb not in calibrated:
            raise ValueError(f"no calibrated MS of {eb} (line {call.line})")
        kwargs = dict(call.kwargs)
        kwargs["vis"] = str(calibrated[eb])
        kwargs["outputvis"] = str(targets / Path(call.kwargs["outputvis"]).name)
        # The calibrated MS holds in DATA what the pipeline's held in CORRECTED_DATA.
        if str(kwargs.get("datacolumn", "")).lower() == "corrected":
            kwargs["datacolumn"] = "data"
        plan.actions.append(Action("mstransform", kwargs, call.line))


def _plan_target_flags(
    plan: ReplayPlan, stages: Sequence[Stage], templates: Mapping[str, str], targets: Path
) -> None:
    """``hifa_flagtargets``: the commands of each EB's template."""
    for call in stage_calls(stages, "hifa_flagtargets"):
        if call.task != "flagdata" or call.kwargs.get("mode") != "list":
            continue
        eb = eb_of(call.kwargs["vis"])
        commands = _flag_commands(templates.get(eb))
        if commands:
            kwargs = {
                "vis": str(targets / Path(call.kwargs["vis"]).name),
                "mode": "list",
                "inpfile": commands,
                "action": "apply",
                "flagbackup": False,
            }
            plan.actions.append(Action("flagdata", kwargs, call.line))
        plan.notes.append(f"{eb}: {len(commands)} target flagging command(s)")


@dataclass
class _SelfcalRouting:
    """Where each ``hif_selfcal`` call runs: on the full-resolution targets until
    the pipeline averaged them into its own folder, then on the averaged copies."""

    targets: Path
    scratch: Path
    averaged: set[str] = field(default_factory=set)

    def where(self, vis: str, task: str, kwargs: Mapping[str, Any]) -> str:
        name = Path(str(vis)).name
        # The flags saved before the averaging are restored on the full-resolution MS.
        before = task == "flagmanager" and kwargs.get("versionname") == "before_hif_selfcal"
        folder = self.scratch if name in self.averaged and not before else self.targets
        return str(folder / name)

    def skipped(self, plan: ReplayPlan, call: CasaCall) -> bool:
        """The calls that make the pipeline's products, not its tables."""
        vis = call.kwargs.get("vis")
        names = [Path(v).name for v in (vis if isinstance(vis, list) else [vis]) if v]
        if any(n.endswith("_line.ms") for n in names):
            return True  # the continuum-subtracted copy, for the pipeline's own cubes
        if call.task == "applycal" and call.kwargs.get("docallib"):
            plan.finally_applied.append(eb_of(str(vis)))
            return True  # the final tables onto the full data, through a library
        if call.task == "tclean" and str(call.kwargs.get("imagename", "")).endswith("_final"):
            return True  # the final image: no table depends on it
        if call.task == "flagdata" and call.kwargs.get("mode") == "summary":
            return True  # a summary changes no flag
        # The summaries of the final application, listed as commands.
        return call.task == "flagdata" and bool(plan.finally_applied)

    def plan(self, plan: ReplayPlan, stages: Sequence[Stage]) -> None:
        for call in stage_calls(stages, "hif_selfcal"):
            if self.skipped(plan, call):
                continue
            if call.task in ("rmtree", "copytree", "move"):
                kwargs = _file_kwargs(call)
            else:
                kwargs = self.routed(call)
            if call.task == "applycal":
                key = (eb_of(str(call.kwargs.get("vis"))), str(call.kwargs.get("field", "")))
                plan.applied[key] = dict(call.kwargs)
            plan.actions.append(Action(call.task, kwargs, call.line, str(self.scratch)))

    def routed(self, call: CasaCall) -> dict[str, Any]:
        kwargs = dict(call.kwargs)
        vis = kwargs.get("vis")
        if call.task == "mstransform":
            name = Path(str(vis)).name
            kwargs["vis"] = str(self.targets / name)
            kwargs["outputvis"] = str(self.scratch / Path(kwargs["outputvis"]).name)
            self.averaged.add(name)
        elif isinstance(vis, list):
            kwargs["vis"] = [self.where(v, call.task, kwargs) for v in vis]
        elif vis:
            kwargs["vis"] = self.where(str(vis), call.task, kwargs)
        if call.task == "tclean":
            kwargs["parallel"] = False
        return kwargs


def _file_kwargs(call: CasaCall) -> dict[str, Any]:
    names = {"rmtree": ("path",), "copytree": ("src", "dst"), "move": ("src", "dst")}[call.task]
    kwargs = dict(zip(names, call.args, strict=False))
    kwargs.update(call.kwargs)
    return kwargs


def apply_recipe(plan: ReplayPlan) -> dict[str, list[dict[str, Any]]]:
    """``{eb: [applycal arguments]}``: the tables of each EB the pipeline applied to
    its data, one entry per field, their names relative to the replay folder."""
    recipe: dict[str, list[dict[str, Any]]] = {}
    for (eb, _), kwargs in sorted(plan.applied.items()):
        if eb not in plan.finally_applied:
            continue
        tables = kwargs.get("gaintable") or []
        tables = [tables] if isinstance(tables, str) else list(tables)
        entry = {
            "field": kwargs.get("field", ""),
            "gaintable": [Path(t).name for t in tables],
            "gainfield": kwargs.get("gainfield", []),
            "interp": kwargs.get("interp", []),
            "spwmap": kwargs.get("spwmap", []),
            "calwt": kwargs.get("calwt", [False]),
            "applymode": kwargs.get("applymode", "calflag"),
        }
        recipe.setdefault(eb, []).append(entry)
    return recipe


# -- running it ---------------------------------------------------------------


def _known(task: Callable[..., Any], kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``kwargs`` less those ``task`` does not take (a newer CASA's), and their names."""
    try:
        parameters = inspect.signature(task).parameters
    except (TypeError, ValueError):
        return kwargs, []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs, []
    dropped = [k for k in kwargs if k not in parameters]
    return {k: v for k, v in kwargs.items() if k in parameters}, dropped


_FILE_TASKS = ("rmtree", "copytree", "move")


def _run_file_task(action: Action, kwargs: dict[str, Any], log: Callable[[str], None]) -> None:
    """Run one of the pipeline's file operations (``rmtree``, ``copytree``, ``move``)."""
    if action.task == "rmtree":
        shutil.rmtree(kwargs["path"], ignore_errors=True)
    elif action.task == "copytree":
        # The pipeline copies image products here; a serial tclean does not write
        # every one a parallel run does (``.gridwt``), and no table depends on them.
        if not Path(kwargs["src"]).exists():
            log(f"line {action.line} copytree: no {kwargs['src']}, skipped")
            return
        shutil.rmtree(kwargs["dst"], ignore_errors=True)
        shutil.copytree(kwargs["src"], kwargs["dst"])
    else:
        shutil.move(kwargs["src"], kwargs["dst"])


def run_actions(actions: Sequence[Action], tasks: Any, log: Callable[[str], None] = print) -> None:
    """Run ``actions`` with the CASA ``tasks`` module, in order."""
    for action in actions:
        kwargs = dict(action.kwargs)
        cwd = action.cwd
        if cwd:
            Path(cwd).mkdir(parents=True, exist_ok=True)
        previous = os.getcwd()
        try:
            if cwd:
                os.chdir(cwd)
            if action.task in _FILE_TASKS:
                _run_file_task(action, kwargs, log)
            else:
                if action.task == "mstransform":
                    shutil.rmtree(kwargs["outputvis"], ignore_errors=True)
                    Path(kwargs["outputvis"]).parent.mkdir(parents=True, exist_ok=True)
                task = getattr(tasks, action.task)
                kwargs, dropped = _known(task, kwargs)
                if dropped:
                    log(f"line {action.line} {action.task}: not in this CASA: {', '.join(dropped)}")
                result = task(**kwargs)
                if action.task == "gaincal" and result is False:
                    raise RuntimeError(f"line {action.line}: gaincal failed")
        finally:
            os.chdir(previous)
        log(f"line {action.line} {action.task} done")


def weblog_flag_templates(member_folder: str | os.PathLike[str], out: Path) -> dict[str, str]:
    """Each EB's target flagging template (``<eb>.flagtargetstemplate.txt``), taken out
    of the member's weblog into ``out``; ``{eb: path}``."""
    found: dict[str, str] = {}
    for weblog in sorted(Path(member_folder).glob("qa/*weblog.tgz")):
        with tarfile.open(weblog) as archive:
            for member in archive.getmembers():
                name = Path(member.name).name
                if not member.isfile() or not name.endswith(".flagtargetstemplate.txt"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                out.mkdir(parents=True, exist_ok=True)
                path = out / name
                path.write_bytes(handle.read())
                found[name.removesuffix(".flagtargetstemplate.txt")] = str(path)
    return found


def casa_commands_log(member_folder: str | os.PathLike[str]) -> Path | None:
    """The member's pipeline CASA commands log that holds ``hif_selfcal``, if any."""
    for path in sorted(Path(member_folder).glob("log/*casa_commands.log")):
        if "# hif_selfcal(" in path.read_text(errors="replace"):
            return path
    return None


def replay_member(
    member_folder: str,
    calibrated: dict[str, str],
    out_dir: str,
    casa_data: str,
    log_file: str,
    keep_work: bool = False,
) -> dict[str, object]:
    """CASA: replay the member's ``hif_selfcal`` and keep its applied tables in ``out_dir``.

    Runs in a CASA child process (``almasim image alma-selfcal``). ``calibrated``:
    ``{eb: calibrated MS}``. Writes ``out_dir/<table>`` and ``out_dir/apply.json``;
    the working folder (``out_dir/work``) is removed unless ``keep_work``.
    """
    from almasim.services.casa import tasks as casa_tasks
    from almasim.services.imaging.selfcal.casa_ops import _configure_casa

    log_path = casa_commands_log(member_folder)
    if log_path is None:
        raise RuntimeError(f"{member_folder}: no hif_selfcal in its casa_commands.log")
    out = Path(out_dir)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    _configure_casa(casa_data, log_file)
    stages = parse_casa_commands(log_path.read_text(errors="replace"))
    templates = weblog_flag_templates(member_folder, work / "flags")
    plan = plan_replay(stages, calibrated, work, templates)
    messages: list[str] = list(plan.notes)

    def log(message: str) -> None:
        messages.append(message)
        with open(log_file, "a") as handle:
            handle.write(f"[alma-selfcal] {message}\n")

    run_actions(plan.actions, casa_tasks(), log)
    recipe = apply_recipe(plan)
    for entries in recipe.values():
        for entry in entries:
            for table in entry["gaintable"]:
                target = out / table
                shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(work / "selfcal" / table, target)
    result: dict[str, object] = {
        "member_folder": str(member_folder),
        "casa_commands_log": str(log_path),
        "calibrated_ms": calibrated,
        "execution_blocks": recipe,
        "not_applied": sorted(set(calibrated) - set(recipe)),
        "notes": plan.notes,
    }
    (out / APPLY_FILE).write_text(json.dumps(result, indent=2) + "\n")
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return result


# -- applying it --------------------------------------------------------------


def find_apply_file(
    member_folder: str | os.PathLike[str] | None,
    member_ous_uid: str,
    output_root: str | os.PathLike[str] | None = None,
) -> Path | None:
    """The replay of a member: in its ``images/alma-selfcal/``, or in
    ``<output_root>/member.<uid>/alma-selfcal/``."""
    folder_name = "member." + member_ous_uid.replace("uid://", "uid___").replace("/", "_")
    candidates = []
    if member_folder:
        candidates.append(Path(member_folder) / "images" / FOLDER / APPLY_FILE)
    if output_root:
        candidates.append(Path(output_root) / folder_name / FOLDER / APPLY_FILE)
    return next((path for path in candidates if path.is_file()), None)


def entries_for(apply_file: str | os.PathLike[str], eb: str, field_name: str) -> list[dict]:
    """The applycal entries of one EB and field of a replay, their tables made absolute."""
    path = Path(apply_file)
    recipe = json.loads(path.read_text())["execution_blocks"].get(eb, [])
    chosen = []
    for entry in recipe:
        fields = {f.strip() for f in str(entry.get("field", "")).split(",") if f.strip()}
        if fields and field_name not in fields:
            continue
        chosen.append({**entry, "gaintable": [str(path.parent / t) for t in entry["gaintable"]]})
    return chosen


def window_spwmap(spwmap: Sequence[Sequence[int]], tables: int, spw: int) -> list[list[int]]:
    """Each table's ``spwmap`` for an MS of the one window ``spw``, renumbered 0: the
    table's window that ``spw`` took in the full MS (itself without a map)."""
    maps = [list(m) for m in spwmap] if spwmap else []
    out = []
    for k in range(tables):
        mapping = maps[k] if k < len(maps) else (maps[0] if len(maps) == 1 else [])
        out.append([int(mapping[spw]) if spw < len(mapping) else int(spw)])
    return out


def apply_selfcal(tasks: Any, vis: str, entries: Sequence[dict], spw: int) -> None:
    """CASA: apply a replay's ``entries`` (``entries_for``) to ``vis``, an MS of the
    one window ``spw`` (renumbered 0) of their execution block."""
    for entry in entries:
        tables = list(entry["gaintable"])
        tasks.applycal(
            vis=vis,
            field=str(entry.get("field", "")),
            gaintable=tables,
            gainfield=entry.get("gainfield") or [],
            interp=entry.get("interp") or [],
            spwmap=window_spwmap(entry.get("spwmap") or [], len(tables), spw),
            calwt=entry.get("calwt", [False]),
            applymode=str(entry.get("applymode") or "calflag"),
            flagbackup=False,
        )
