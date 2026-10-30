"""Save and load resolved products as CSV. (MADIV-28 T6.9)"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .models import DataProduct

logger = logging.getLogger(__name__)


def products_to_dataframe(products: list[DataProduct]) -> pd.DataFrame:
    """Convert DataProduct rows into a DataFrame."""
    rows = []
    for product in products:
        row = asdict(product)
        row["size_mb"] = product.size_mb
        rows.append(row)
    return pd.DataFrame(rows)


def save_products_csv(products: list[DataProduct], output_path: Path | str) -> str:
    """Persist resolved DataLink products as CSV."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    products_to_dataframe(products).to_csv(output_path, index=False)
    return str(output_path)


def load_products_csv(input_path: Path | str) -> list[DataProduct]:
    """Load previously saved DataProduct rows from CSV."""
    input_path = Path(input_path).expanduser().resolve()
    # Blank fields stay "": pandas would read them as NaN, which str() turns into "nan".
    dataframe = pd.read_csv(input_path, keep_default_na=False)
    products = []
    for row in dataframe.to_dict(orient="records"):
        products.append(
            DataProduct(
                access_url=str(row["access_url"]),
                uid=str(row["uid"]),
                filename=str(row["filename"]),
                content_length=int(row.get("content_length", 0) or 0),
                content_type=str(row.get("content_type", "") or ""),
                product_type=str(row.get("product_type", "other") or "other"),
                semantics=str(row.get("semantics", "") or ""),
                path=str(row.get("path", "") or ""),
            )
        )
    return products
