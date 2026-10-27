"""The EB-time test: library image noise over real, before (MOUS Int.Time) and
after (the real execution block's time on source), against the prediction
old x sqrt(time ratio).

    python libeb_compare.py PICK.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NOISE = Path("/SKAO_data/cycle11/MADIV-28/noise")


def main(pick: str) -> None:
    p = pd.read_csv(pick)
    names = dict(line.split() for line in open(NOISE / "diag" / "index-names.txt"))
    out = []
    for _, r in p.iterrows():
        name = names[str(int(r.window))]
        real = json.loads((NOISE / "iter-f0-control" / "metrics" / f"{name}.json").read_text())
        path = NOISE / "iter-lib-eb" / "metrics" / f"{name}.json"
        if not path.exists():
            out.append({"window": int(r.window), "status": "missing"})
            continue
        eb = json.loads(path.read_text())
        out.append({
            "window": int(r.window), "time": r.time,
            "cont_before": r.image_cont, "cont_predicted": r.predicted,
            "cont_after": eb["cont.sigma"] / real["cont.sigma"],
            "cube_before": r.image_cube, "cube_predicted": r.predicted_cube,
            "cube_after": eb["cube.sigma_median"] / real["cube.sigma_median"],
        })
    d = pd.DataFrame(out)
    print(d.round(3).to_string(index=False))
    ok = d.dropna(subset=["cont_after"]) if "cont_after" in d else d
    for k in ("cont", "cube"):
        print(f"{k}: median before {ok[f'{k}_before'].median():.3f}, predicted {ok[f'{k}_predicted'].median():.3f}, "
              f"after {ok[f'{k}_after'].median():.3f}; after/predicted {np.median(ok[f'{k}_after'] / ok[f'{k}_predicted']):.3f}; "
              f"within 10% of real after: {int((np.abs(ok[f'{k}_after'] - 1) <= 0.1).sum())} of {len(ok)}")


if __name__ == "__main__":
    main(sys.argv[1])
