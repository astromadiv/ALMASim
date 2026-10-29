"""Released noise models and how a model is named on the command line (MADIV-28).

Released models are the package's ``models/<name>-<version>.json`` files. A
released file is never edited: a change is the next version, through a PR.

A site can keep its own models beside them: the same ``<name>-<version>.json``
files in the folder ``ALMASIM_NOISE_MODELS`` names, else the profile's
``[tools] noise_models`` (the shared install's ``noise-models/``, where the
console releases the team's models). Jobs get the profile in their
environment, so they resolve the same names. A site model may not take a
package model's name: ``alma`` is always the released one.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from .model import NoiseModel

logger = logging.getLogger(__name__)

SITE_ENV = "ALMASIM_NOISE_MODELS"
PACKAGE, SITE = "package", "site"
_RELEASED = re.compile(r"^(?P<name>[a-z][a-z0-9_-]*)-(?P<version>[1-9][0-9]*)\.json$")
_SPEC = re.compile(r"^(?P<name>[a-z][a-z0-9_-]*)(?:@(?P<version>[1-9][0-9]*))?$")
NONE = "none"


@cache
def packaged() -> dict[str, dict[int, NoiseModel]]:
    """The models released with the package, by name and version."""
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


def site_models_dir(environ: Mapping[str, str] | None = None) -> Path | None:
    """The site's models folder: ``ALMASIM_NOISE_MODELS``, else the profile's; or None."""
    env = os.environ if environ is None else environ
    if env.get(SITE_ENV):
        return Path(env[SITE_ENV]).expanduser()
    from almasim.jobs.profile import ProfileError, load_profile

    try:
        return load_profile(environ=env).noise_models
    except ProfileError as exc:
        logger.warning("noise models: cannot read the profile: %s", exc)
        return None


def site_models(directory: Path | None = None) -> dict[str, dict[int, NoiseModel]]:
    """The site folder's models, by name and version. Read afresh each time: a
    release adds one while the console runs. A file that does not load, or
    takes a package model's name, is skipped with a warning, so one bad file
    does not hide the others."""
    directory = site_models_dir() if directory is None else directory
    models: dict[str, dict[int, NoiseModel]] = {}
    if directory is None or not directory.is_dir():
        return models
    for entry in sorted(directory.iterdir()):
        match = _RELEASED.match(entry.name)
        if not match:
            continue
        if match["name"] in packaged():
            logger.warning("noise model %s ignored: %r is a package model", entry, match["name"])
            continue
        try:
            model = NoiseModel.model_validate_json(entry.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("noise model %s ignored: %s", entry, exc)
            continue
        if (model.name, model.version) != (match["name"], int(match["version"])):
            logger.warning(
                "noise model %s ignored: it holds %s@%s", entry, model.name, model.version
            )
            continue
        models.setdefault(model.name, {})[model.version] = model
    return models


def released() -> dict[str, dict[int, NoiseModel]]:
    """Every released model by name and version: the package's, then the site's."""
    return {**{name: dict(v) for name, v in packaged().items()}, **site_models()}


def model_source(name: str) -> str:
    """Where a released model comes from: ``package`` or ``site``."""
    return PACKAGE if name in packaged() else SITE


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
