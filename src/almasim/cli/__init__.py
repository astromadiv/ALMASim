"""ALMASim CLI entrypoint and lazy command registration.

The command groups live in this package (``image``, ``products``, ...); the
old ``almasim.cli_<group>`` aliases were removed in MADIV-28 T8.4.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import click
import typer

app = typer.Typer(
    help="ALMASim command-line interface.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _setup() -> None:
    """ALMASim command-line interface."""
    from ._shared import configure_logging

    configure_logging()


def _invoke_click_command(command: click.Command, *, args: list[str], prog_name: str) -> None:
    """Invoke a click command while preserving Typer-style exit behavior."""
    try:
        result = command.main(
            args=args,
            prog_name=prog_name,
            standalone_mode=False,
        )
    except click.exceptions.Exit as exc:
        raise typer.Exit(code=exc.exit_code) from exc

    if isinstance(result, int) and result != 0:
        raise typer.Exit(code=result)


def _forward_typer_subapp(
    module_name: str,
    app_name: str,
    *,
    ctx: typer.Context,
    prog_name: str,
) -> None:
    """Lazy-load a Typer sub-app and forward CLI args to it."""
    from ._options import deprecated_flag_warnings

    module = import_module(module_name, package=__package__)
    subapp = getattr(module, app_name)
    command = typer.main.get_command(subapp)
    args = list(ctx.args) or ["--help"]
    for warning in deprecated_flag_warnings(args):
        typer.echo(warning, err=True)

    # Typer may emit a plain Command for single-command apps; in that case,
    # users still type an explicit subcommand token (e.g. `simulation run ...`).
    if not isinstance(command, click.MultiCommand) and args and args[0] == command.name:
        args = args[1:]

    _invoke_click_command(command, args=args, prog_name=prog_name)


def _forward_clean_command(*, ctx: typer.Context) -> None:
    """Lazy-load and invoke the WSClean passthrough command."""
    module = import_module(".clean", package=__package__)
    clean_app = typer.Typer(add_completion=False)
    clean_app.command(
        "clean",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )(module.clean_command)
    command = typer.main.get_command(clean_app)
    _invoke_click_command(command, args=list(ctx.args) or ["--help"], prog_name="almasim clean")


# (command, module, Typer app, help): each group's module is imported only
# when the group runs, so ``almasim --help`` stays fast.
_GROUPS = (
    ("metadata", ".metadata", "metadata_app", "Metadata query commands."),
    ("products", ".products", "products_app", "Data product resolution and download commands."),
    ("simulation", ".simulation", "simulation_app", "Simulation workflow commands."),
    ("visibilities", ".visibilities", "visibilities_app", "Visibility processing commands."),
    ("predict", ".predict", "predict_app", "Model prediction commands."),
    ("image", ".image", "image_app", "Data product batch imaging."),
    ("preview", ".preview", "preview_app", "Previews of products, written as FITS."),
    ("noise", ".noise", "noise_app", "Noise models: released versions, components, schema."),
    ("jobs", ".jobs", "jobs_app", "List, follow and control submitted runs."),
)

_FORWARDED = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}


def _register_group(name: str, module_name: str, app_name: str, help_text: str) -> None:
    def forward(ctx: typer.Context) -> None:
        _forward_typer_subapp(module_name, app_name, ctx=ctx, prog_name=f"almasim {name}")

    forward.__name__ = f"{name}_command"
    forward.__doc__ = f"Run {name} subcommands."
    app.command(name, help=help_text, context_settings=_FORWARDED)(forward)


for _group in _GROUPS:
    _register_group(*_group)


@app.command(
    "clean",
    help="Run WSClean and forward all extra arguments unchanged.",
    context_settings=_FORWARDED,
)
def clean_command_proxy(ctx: typer.Context) -> None:
    """Run clean passthrough command."""
    _forward_clean_command(ctx=ctx)


@app.command("doctor", help="Check that the job profile works on this machine.")
def doctor_command(
    slurm: bool = typer.Option(
        False, "--slurm", help="Run the checks on a compute node (with srun), as a job sees them."
    ),
    site_profile: Path | None = typer.Option(
        None, "--site-profile", help="Site profile (default: $ALMASIM_SITE_PROFILE)."
    ),
    profile: Path | None = typer.Option(
        None,
        "--profile",
        help="User profile (default: $ALMASIM_PROFILE, else ~/.config/almasim/almasim.toml).",
    ),
) -> None:
    """Check the runs root, Python, CASA, WSClean and Slurm the profile names."""
    from .doctor import doctor

    doctor(slurm, site_profile, profile)


@app.command(
    "update-shared",
    help=(
        "Update the team's shared install (the one this almasim runs from) to a git ref, "
        "as a Slurm job: clone, build, reinstall, then doctor."
    ),
)
def update_shared_command(
    ref: str = typer.Option("main", "--ref", help="Branch, tag or commit to install."),
    prefix: Path | None = typer.Option(
        None, "--prefix", help="The install (default: the one this almasim runs from)."
    ),
    repo: str | None = typer.Option(
        None, "--repo", help="Git URL to clone (default: the one the install recorded)."
    ),
    template: str | None = typer.Option(
        None,
        "--template",
        help="Site profile template, relative to the repo (default: the recorded one).",
    ),
    force: bool = typer.Option(
        False, "--force", help="Update even while runs are not finished (they may break)."
    ),
) -> None:
    """Rebuild the shared install from ``ref`` with the recorded repo and template."""
    from .update_shared import update_shared

    update_shared(ref, prefix, repo, template, force)


@app.command(
    "submit",
    help=(
        "Run an almasim command as a job on Slurm (or on this node, where the profile says "
        "so). Options before the command are the job's; the rest is the "
        "command, e.g. `almasim submit --cpus 8 image image-from-ms params.csv out --cube`."
    ),
    context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False},
)
def submit_command(
    command: list[str] = typer.Argument(..., help="The almasim command line to run."),
    partition: str | None = typer.Option(None, "--partition", help="Slurm partition."),
    account: str | None = typer.Option(None, "--account", help="Slurm account."),
    cpus: int | None = typer.Option(None, "--cpus", min=1, help="CPUs for the job."),
    memory: str | None = typer.Option(None, "--memory", help="Memory for the job, e.g. 16G."),
    time_limit: str | None = typer.Option(
        None, "--time", help="Time limit, e.g. 02:00:00 or 2-00:00:00."
    ),
    title: str = typer.Option("", "--title", help="A name for the run."),
    after: list[str] | None = typer.Option(
        None, "--after", help="Start only when this run has succeeded (its id). Repeatable."
    ),
    wait: bool = typer.Option(
        False, "--wait", help="Wait for the run to end; exit 1 unless it succeeded."
    ),
    profile: Path | None = typer.Option(
        None,
        "--profile",
        help="User profile (default: $ALMASIM_PROFILE, else ~/.config/almasim/almasim.toml).",
    ),
) -> None:
    """Submit ``almasim <command>`` with the profile's defaults and these overrides."""
    from .jobs import submit

    submit(
        command,
        partition=partition,
        account=account,
        cpus=cpus,
        memory=memory,
        time_limit=time_limit,
        title=title,
        wait=wait,
        user_profile=profile,
        after=after,
    )


@app.command("serve", help="Run the web console on this machine (needs almasim[web]).")
def serve_command(
    port: int | None = typer.Option(
        None,
        "--port",
        min=1024,
        max=65535,
        help=(
            "Listen on 127.0.0.1:<port> instead of a private socket in ~/.almasim"
            " (in a container, ALMASIM_SERVE_HOST sets the address)."
        ),
    ),
    base_path: str = typer.Option(
        "",
        "--base-path",
        envvar="ALMASIM_BASE_PATH",
        help=(
            "Serve under this URL prefix, for a reverse proxy that passes it on"
            " (Open OnDemand's /node/<host>/<port>)."
        ),
    ),
    managed: bool = typer.Option(
        False,
        "--managed",
        help=(
            "Started and stopped by a portal (Open OnDemand): needs --port and keeps"
            " no record in ~/.almasim, so --status and --stop do not see it."
        ),
    ),
    status: bool = typer.Option(False, "--status", help="Show the running console's link."),
    stop: bool = typer.Option(False, "--stop", help="Stop the running console."),
) -> None:
    """Serve the console; print the ssh tunnel line and the link with its access token."""
    from .serve import serve

    serve(port, status, stop, base_path, managed)


def _raise_open_file_limit() -> None:
    """Lift the soft open-file limit to the hard one.

    WSClean's cube reordering keeps three files open per output channel (data,
    weight, model), so a 2048-channel cube needs over 6000 at once. A console
    started from a shell with the usual soft limit of 1024 passed it on to every
    Slurm job it submitted (PropagateResourceLimits), and those cubes failed
    with "Error opening temporary meta file" (2026-10-28). Raising it here
    covers this process, WSClean and every job submitted from it.
    """
    try:
        import resource
    except ImportError:  # Windows: no rlimits
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # An unlimited hard limit cannot be the soft one for open files on Linux.
    target = hard if hard != resource.RLIM_INFINITY else 1 << 20
    if soft != resource.RLIM_INFINITY and soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError):
            pass  # keep the inherited limit rather than refuse to start


def main() -> None:
    """Run the ALMASim CLI application."""
    _raise_open_file_limit()
    app()
