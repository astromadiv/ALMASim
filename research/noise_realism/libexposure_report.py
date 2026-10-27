"""The library's image noise over the real, split into what it observes.

    python libexposure_report.py EXPOSURE.csv

image ratio = natural-noise ratio x imaging-factor ratio, and the natural-noise
ratio = per-visibility noise (scaled to one second) x sqrt(real/library
visibility-seconds). Int.Time is the archive's MOUS total; the real window is
one execution block (n_eb per MOUS from the products table).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def main(path: str) -> None:
    d = pd.read_csv(path)
    p = pd.read_csv("/SKAO_data/cycle11/metadata/resolved_cycle11_products.csv")
    asdm = p[p.filename.str.contains(".asdm.sdm", regex=False)]
    rows = pd.read_csv("/SKAO_data/cycle11/MADIV-28/noise/libsim/rows.csv", usecols=["window", "member_ous_uid"])
    d = d.merge(rows, on="window")
    d["n_eb"] = d.member_ous_uid.map(asdm.groupby("uid").filename.nunique())
    r = pd.DataFrame({"window": d.window})
    r["image_cont"] = d["lib:cont_sigma"] / d["real:cont_sigma"]
    r["image_cube"] = d["lib:cube_sigma"] / d["real:cube_sigma"]
    r["natural"] = d["lib:sigma_natural"] / d["real:sigma_natural"]
    r["imaging"] = r.image_cont / r.natural
    r["time"] = d["lib:time_on_source"] / d["real:time_on_source"]
    r["inttime_over_neb"] = d["Int.Time"] / d.n_eb / d["real:time_on_source"]
    r["antennas"] = d["lib:antennas"] / d["real:antennas"]
    r["baselines"] = (d["lib:antennas"] * (d["lib:antennas"] - 1)) / (d["real:antennas"] * (d["real:antennas"] - 1))
    r["bandwidth"] = d["lib:bandwidth"] / d["real:bandwidth"]
    r["flag_real"] = d["real:flagged"]
    # noise per visibility for 1 s and one channel width: sigma sqrt(t dnu)
    unit = lambda k: d[f"{k}:sigma_vis"] * np.sqrt(d[f"{k}:t_int"] * d[f"{k}:chan_width"])  # noqa: E731
    r["radiometer"] = unit("lib") / unit("real")
    # what the natural noise ratio would be from data volume alone
    vol = lambda k: d[f"{k}:vis"] * d[f"{k}:t_int"] * d[f"{k}:chan_width"]  # noqa: E731
    r["volume"] = np.sqrt(vol("real") / vol("lib"))
    r["natural_check"] = r.radiometer * r.volume
    r["n_eb"] = d.n_eb
    pd.set_option("display.width", 250)
    q = r.drop(columns="window").quantile([0.16, 0.5, 0.84]).T
    q.columns = ["q16", "median", "q84"]
    print(q.round(3).to_string())
    for col in ("image_cont", "natural", "time"):
        print(col, "within 10% of 1:", int((np.abs(r[col] - 1) <= 0.1).sum()), "of", int(r[col].notna().sum()))
    print("corr(log image_cont, log natural)", np.corrcoef(np.log(r.image_cont), np.log(r.natural))[0, 1].round(3))
    print("corr(log natural, -0.5 log time)", np.corrcoef(np.log(r.natural), -0.5 * np.log(r.time))[0, 1].round(3))
    r.to_csv(path.replace(".csv", "-ratios.csv"), index=False)
    print(r.sort_values("image_cont").round(3).to_string(index=False, max_rows=30))


if __name__ == "__main__":
    main(sys.argv[1])
