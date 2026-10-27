"""Noise models: what is released, what each component does, the file schema (MADIV-28).

``noise show`` lists the released models, or prints one (a released name, a
candidate file) with the components it applies; ``noise components`` lists
the components a model can hold and their parameters; ``noise schema``
writes the JSON Schema model files validate against.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..services.noise_model.components import NoiseComponent
from ._shared import fail

noise_app = typer.Typer(
    help="Noise models: released versions, components, schema.",
    no_args_is_help=True,
)


@noise_app.command("show")
def show(
    model: str | None = typer.Argument(
        None,
        help=(
            "A released model ('alma', 'alma@1'), or a JSON file; omitted: list the "
            "released models."
        ),
    ),
    only: str | None = typer.Option(
        None, "--noise-only", help="Show the model with only these components (comma-separated)."
    ),
    without: str | None = typer.Option(
        None, "--noise-without", help="Show the model without these components (comma-separated)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the model's JSON."),
) -> None:
    """List the released noise models, or show one."""
    from ..services.noise_model import released, resolve_noise_model

    if model is None:
        for name, versions in sorted(released().items()):
            newest = versions[max(versions)]
            typer.echo(
                f"{name}: versions {', '.join(map(str, sorted(versions)))}; "
                f"newest applies {', '.join(newest.enabled()) or 'nothing'}"
            )
        return
    try:
        resolved = resolve_noise_model(model)
        if resolved is None:
            typer.echo("none: applies nothing")
            return
        resolved = resolved.select(
            only=[only] if only is not None else None, without=[without] if without else None
        )
    except (ValueError, TypeError) as exc:
        fail(f"noise model {model!r}: {exc}")
    if as_json:
        typer.echo(resolved.model_dump_json(indent=2))
        return
    typer.echo(f"{resolved.label()}  sha256 {resolved.sha256()}")
    if resolved.description:
        typer.echo(resolved.description)
    if resolved.provenance is not None:
        p = resolved.provenance
        typer.echo(f"provenance: {p.method}" + (f"; {p.data}" if p.data else ""))
    typer.echo(f"settings: {_values(resolved.settings.model_dump())}")
    for name, component in resolved.components.items():
        params = component.model_dump(exclude={"enabled"})
        state = "on " if component.enabled else "off"
        typer.echo(f"  [{state}] {name}" + (f": {_values(params)}" if params else ""))


@noise_app.command("components")
def components() -> None:
    """The components a noise model can hold, and their parameters."""
    from ..services.noise_model import Components

    for name, info in Components.model_fields.items():
        cls = _component_class(info.annotation)
        typer.echo(f"{name}: {' '.join((cls.__doc__ or '').split())}")
        for field, spec in cls.model_fields.items():
            if field == "enabled":
                continue
            default = "required" if spec.is_required() else f"default {spec.default}"
            typer.echo(f"    {field} ({default}): {spec.description or ''}")


@noise_app.command("schema")
def schema(
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write it here (default: print)."
    ),
) -> None:
    """The JSON Schema of a noise model file."""
    from ..services.noise_model.schema import noise_model_schema

    text = json.dumps(noise_model_schema(), indent=2) + "\n"
    if output is None:
        typer.echo(text, nl=False)
    else:
        output.write_text(text)
        typer.echo(f"wrote {output}")


def _component_class(annotation: object) -> type[NoiseComponent]:
    """The component class of a ``Components`` field (``X | None``)."""
    for arg in getattr(annotation, "__args__", ()):
        if isinstance(arg, type) and issubclass(arg, NoiseComponent):
            return arg
    raise TypeError(f"not a component field: {annotation!r}")


def _values(values: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in values.items())
