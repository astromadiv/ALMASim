"""Release a validated candidate as the next version of a noise model (MADIV-28).

A candidate (``noise aggregate``'s, or one written by hand) becomes
``models/<name>-<n+1>.json`` only with a comparison of that very model
(``noise validate``'s comparison.json, matched by the model's SHA-256) that
passed its criteria. The file is written into the working tree; the pull
request that adds it, and with it moves ``<name>`` to the new version, stays
with a person.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..noise_model import NoiseModel, Provenance
from .compare import Comparison

MODELS_DIR = Path(__file__).resolve().parents[1] / "noise_model" / "models"


def promote(
    candidate: NoiseModel,
    comparison: Comparison,
    *,
    name: str | None = None,
    models_dir: Path = MODELS_DIR,
) -> Path:
    """Write the candidate as the next released version; its path. A
    ValueError when the comparison is of another model, or did not pass."""
    sha = candidate.sha256()
    if comparison.model.get("sha256") != sha:
        raise ValueError(
            f"the comparison is of {comparison.model.get('label', 'another model')}, "
            f"not {candidate.label()} (sha256 {sha[:12]})"
        )
    if not comparison.passed:
        failed = "; ".join(c.name for c in comparison.checks if not c.passed) or "no checks"
        raise ValueError(f"{candidate.label()} did not pass its validation: {failed}")
    name = name or candidate.name.removesuffix("-candidate")
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError(f"not a model name: {name!r}")
    versions = [
        int(m["version"])
        for path in models_dir.glob(f"{name}-*.json")
        if (m := re.fullmatch(rf"{re.escape(name)}-(?P<version>[1-9][0-9]*)\.json", path.name))
    ]
    version = max(versions, default=0) + 1
    target = models_dir / f"{name}-{version}.json"
    note = (
        f"Validated on {comparison.windows} windows: {candidate.label()} "
        f"(sha256 {sha[:12]}) passed every check."
    )
    provenance = candidate.provenance or Provenance(method="hand-set")
    notes = f"{provenance.notes} {note}" if provenance.notes else note
    released = candidate.model_copy(
        update={
            "name": name,
            "version": version,
            "provenance": provenance.model_copy(update={"notes": notes}),
        }
    )
    released = NoiseModel.model_validate_json(released.model_dump_json())
    models_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(released.model_dump_json(indent=2) + "\n")
    return target
