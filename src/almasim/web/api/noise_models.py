"""The noise models a simulation can use, for the Simulate page's picker.

Every version of every released model: the package's and the site folder's
(the profile's ``[tools] noise_models``, where the team releases its own), each
with its components and where its values came from. The folder is read on
every request, so a model released while the console runs shows at once.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from almasim.services.noise_model import NoiseModel, packaged, site_models
from almasim.services.noise_model.components import describe_component
from almasim.services.noise_model.registry import PACKAGE, SITE, SITE_ENV

router = APIRouter(prefix="/api/v2/noise-models", tags=["noise models"])

# What `noise promote` adds to a released model's provenance notes.
_VALIDATED = re.compile(r"\(sha256 (?P<sha>[0-9a-f]{12})\) passed every check")


class Component(BaseModel):
    name: str
    enabled: bool
    description: str


class ModelProvenance(BaseModel):
    method: str
    data: str | None
    windows: int | None
    date: str | None
    notes: str | None
    # The candidate's SHA-256 (first 12) whose validation passed, from `noise promote`.
    validated: str | None


class NoiseModelInfo(BaseModel):
    spec: str  # what --noise-model takes: name@version
    name: str
    version: int
    newest: bool  # the version the bare name selects
    source: Literal["package", "site"]
    sha256: str
    description: str
    components: list[Component]
    provenance: ModelProvenance | None


class NoiseModels(BaseModel):
    models: list[NoiseModelInfo]
    site_folder: str | None


def _site_folder(request: Request) -> Path | None:
    if os.environ.get(SITE_ENV):
        return Path(os.environ[SITE_ENV]).expanduser()
    folder: Path | None = request.app.state.profile.noise_models
    return folder


def _info(model: NoiseModel, *, newest: bool, source: str) -> dict[str, Any]:
    p = model.provenance
    validated = _VALIDATED.search(p.notes or "") if p is not None else None
    return {
        "spec": f"{model.name}@{model.version}",
        "name": model.name,
        "version": model.version,
        "newest": newest,
        "source": source,
        "sha256": model.sha256(),
        "description": model.description,
        "components": [
            {"name": n, "enabled": c.enabled, "description": describe_component(n)}
            for n, c in model.components.items()
        ],
        "provenance": None
        if p is None
        else {
            "method": p.method,
            "data": p.data,
            "windows": p.windows,
            "date": p.date,
            "notes": p.notes,
            "validated": validated["sha"] if validated else None,
        },
    }


@router.get("", response_model=NoiseModels)
def list_noise_models(request: Request) -> dict[str, Any]:
    """Every released noise model and version, the package's first, newest version first."""
    folder = _site_folder(request)
    models: list[dict[str, Any]] = []
    for source, found in ((PACKAGE, packaged()), (SITE, site_models(folder) if folder else {})):
        for name in sorted(found):
            versions = found[name]
            for version in sorted(versions, reverse=True):
                models.append(
                    _info(versions[version], newest=version == max(versions), source=source)
                )
    return {"models": models, "site_folder": str(folder) if folder else None}
