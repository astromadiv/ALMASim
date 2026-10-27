"""Released noise models and how a model is named on the command line (MADIV-28).

Released models are the package's ``models/<name>-<version>.json`` files. A
released file is never edited: a change is the next version, through a PR.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from .model import NoiseModel

_RELEASED = re.compile(r"^(?P<name>[a-z][a-z0-9_-]*)-(?P<version>[1-9][0-9]*)\.json$")
_SPEC = re.compile(r"^(?P<name>[a-z][a-z0-9_-]*)(?:@(?P<version>[1-9][0-9]*))?$")
NONE = "none"


@cache
def released() -> dict[str, dict[int, NoiseModel]]:
    """Every released model, by name and version."""
    models: dict[str, dict[int, NoiseModel]] = {}
    for entry in resources.files(__package__).joinpath("models").iterdir():
        match = _RELEASED.match(entry.name)
        if not match:
            continue
        model = NoiseModel.model_validate_json(entry.read_text())
        if (model.name, model.version) != (match["name"], int(match["version"])):
            raise ValueError(f"noise model file {entry.name} holds {model.name}@{model.version}")
        models.setdefault(model.name, {})[model.version] = model
    return models


def released_model(name: str, version: int | None = None) -> NoiseModel:
    """A released model; its newest version when ``version`` is None."""
    versions = released().get(name)
    if not versions:
        raise ValueError(
            f"no released noise model {name!r}; released: {', '.join(released()) or 'none'}"
        )
    if version is None:
        return versions[max(versions)]
    if version not in versions:
        raise ValueError(
            f"noise model {name!r} has no version {version}; "
            f"it has {', '.join(map(str, sorted(versions)))}"
        )
    return versions[version]


def resolve_noise_model(spec: NoiseModel | str | Mapping[str, Any] | None) -> NoiseModel | None:
    """The model ``spec`` names: a released model (``alma``, ``alma@1``),
    ``none``, a JSON object or a path to a JSON file (a model, or the flat
    CorruptionModel fields of earlier versions), or a model. None for none."""
    if spec is None or isinstance(spec, NoiseModel):
        return spec
    if isinstance(spec, Mapping):
        return _from_mapping(spec, name="custom")
    text = str(spec).strip()
    if not text or text == NONE:
        return None
    match = _SPEC.match(text)
    if match and match["name"] in released():
        return released_model(match["name"], int(match["version"]) if match["version"] else None)
    if text.startswith("{"):
        return _from_mapping(json.loads(text), name="custom")
    path = Path(text).expanduser()
    if not path.is_file():
        raise ValueError(
            f"noise model {text!r}: not a released model ({', '.join(released())}, {NONE}), "
            "a JSON object or a JSON file"
        )
    stem = re.sub(r"[^a-z0-9_-]", "-", path.stem.lower()).lstrip("0123456789_-") or "custom"
    return _from_mapping(json.loads(path.read_text()), name=stem)


def _from_mapping(values: Mapping[str, Any], *, name: str) -> NoiseModel | None:
    """A model from its JSON form, or from the flat CorruptionModel fields."""
    if not values:
        return None
    if not set(values) & set(NoiseModel.model_fields):
        # The flat CorruptionModel fields; an unknown one fails there, by name.
        model = NoiseModel.from_corruption_fields(values, name=name)
        return model if model.engine_fields() is not None else None
    return NoiseModel.model_validate(values)
