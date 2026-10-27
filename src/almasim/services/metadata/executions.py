"""How many execution blocks observed each metadata row's MOUS (MADIV-28).

The archive's ``Int.Time`` is the MOUS's total time on source, summed over its
execution blocks; ObsCore does not say how many there were (its ``asdm_uid``
names one per field). DataLink lists every raw ASDM of a MOUS, one per block:
the simulator observes ``Int.Time`` in that many blocks (``n_executions``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from ..download.datalink import count_executions, resolve_products

COLUMN = "n_executions"


def add_execution_counts(
    metadata: pd.DataFrame,
    *,
    resolve: Callable[[Iterable[str]], list[Any]] = resolve_products,
) -> pd.DataFrame:
    """``metadata`` with ``n_executions``: each row's MOUS's execution blocks,
    from DataLink; empty where none were found."""
    if "member_ous_uid" not in metadata.columns:
        raise KeyError("metadata has no member_ous_uid column")
    uids = metadata["member_ous_uid"].dropna().astype(str).unique().tolist()
    counts = count_executions(resolve(uids)) if uids else {}
    out = metadata.copy()
    out[COLUMN] = out["member_ous_uid"].map(counts).astype("Int64")
    return out
