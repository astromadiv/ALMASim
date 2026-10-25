"""ALMASim's own simulations of the MADIV-16 windows, with the corruptions the
counterparts measured (the "alma" preset), to compare with the real data.

    python libsim.py rows OUT.csv              each window's archive row (cycle11.csv,
                                               by its OUS), its Freq moved to the window
    python libsim.py run I ROWS.csv OUT_DIR    window I's simulation, written as an MS

The run is what `almasim simulation run` does with that row (the redshift of
its catalogue line, a point source, the row's array and integration time),
with the MS written (the CLI leaves it out) and the window's own channel count
(at most MAX_CHANNELS).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window import name, subset  # noqa: E402

META = Path("/SKAO_data/cycle11/metadata")
BASE = Path("/SKAO_data/cycle11/MADIV-28/noise/base")
LIBRARY = Path(os.environ.get("ALMASIM_SRC", "/home1/mdelliveneri/ALMASim-atm/src/almasim"))
MAX_CHANNELS = 960


def rows(out: Path) -> None:
    """One archive row per window: its OUS's row, Freq at the window's centre."""
    from casacore.tables import table

    meta = pd.read_csv(META / "cycle11.csv")
    products = pd.read_csv(META / "resolved_cycle11_products.csv", usecols=["uid", "filename"])
    out_rows = []
    for i, row in subset().iterrows():
        eb = Path(row.ms).name.split(".ms")[0].removeprefix("uid___")
        ous = products[products.filename.str.contains(eb, regex=False)].uid.unique()
        match = meta[meta.member_ous_uid.isin(ous)]
        if match.empty:
            print(f"{i}: no archive row for {eb}")
            continue
        with table(str(BASE / f"{name(row)}.ms" / "SPECTRAL_WINDOW"), ack=False) as t:
            chan = np.asarray(t.getcell("CHAN_FREQ", 0))
        record = match.iloc[0].to_dict()
        record.update(
            window=int(i),
            window_name=name(row),
            Freq=float(np.mean(chan)) / 1e9,
            n_channels=int(min(len(chan), MAX_CHANNELS)),
        )
        out_rows.append(record)
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f"{len(out_rows)} windows -> {out}")


def run(i: int, rows_csv: Path, out_dir: Path) -> None:
    from almasim.services import astro
    from almasim.services.astro.spectral import with_redshifts
    from almasim.services.compute import create_backend
    from almasim.services.simulation import (
        SimulationParams,
        export_results,
        generate_clean_cube,
        simulate_observation,
    )

    table = pd.read_csv(rows_csv)
    picked = table[table.window == i]
    if picked.empty:
        raise SystemExit(f"window {i} has no row in {rows_csv}")
    rest_frequency, _ = astro.get_line_info(LIBRARY)
    row = with_redshifts(picked.iloc[[0]].reset_index(drop=True), rest_frequency).iloc[0]
    run_dir = out_dir / f"w{i}"
    run_dir.mkdir(parents=True, exist_ok=True)
    params = SimulationParams.from_metadata_row(
        row,
        idx=i,
        main_dir=LIBRARY,
        output_dir=run_dir,
        tng_dir=run_dir / "tng",
        galaxy_zoo_dir=run_dir / "galaxy_zoo",
        hubble_dir=run_dir / "hubble",
        project_name=f"w{i}",
        source_type="point",
        save_mode="fits",
        persist=True,
        ml_dataset_path=None,
        n_pix=256,
        n_channels=int(row.n_channels),
        seed=1000 + i,
        corruptions="alma",
        ms_export=True,
        ms_save_mode="msv2",
    )
    with create_backend("sync") as backend:
        clean = generate_clean_cube(params, logger=print, compute_backend=backend)
        results = simulate_observation(clean, compute_backend=backend)
        exported = export_results(params, clean, results, logger=print)
    print("MS", exported.get("ms_path"))


if __name__ == "__main__":
    if sys.argv[1] == "rows":
        rows(Path(sys.argv[2]))
    else:
        run(int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
