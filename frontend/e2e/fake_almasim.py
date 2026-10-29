"""A stand-in for the almasim CLI, for the console's end-to-end tests (U3.9).

The test console runs it in place of `almasim` (ALMASIM_JOB_COMMAND), so a
run takes a second instead of hours and needs neither the ALMA archive nor
CASA nor WSClean: each command writes what the real one would leave for the
pages to read (CSV files, task folders with their markers, small FITS
images, quality.json), and logs a few lines.

`doctor --slurm` runs for a minute, for the cancel test.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
from astropy.io import fits


def option(argv: list[str], flag: str, default: str | None = None) -> str | None:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def positional(argv: list[str], skip: int) -> list[str]:
    """The arguments after the command words that are not options or their values."""
    out, rest, i = [], argv[skip:], 0
    while i < len(rest):
        if rest[i].startswith("--"):
            i += 1 if "=" in rest[i] or rest[i] in BOOLEAN_FLAGS else 2
            continue
        out.append(rest[i])
        i += 1
    return out


BOOLEAN_FLAGS = {"--yes", "--cube", "--slurm", "--extract-tar", "--unpack-ms",
                 "--generate-calibrated-visibilities", "--persist-standard-outputs",
                 "--skip-casa-data-update", "--all-spws", "--allow-partial"}  # fmt: skip


def say(*lines: str, pause: float = 0.2) -> None:
    for line in lines:
        print(line, flush=True)
        time.sleep(pause)


def write_fits(path: Path, channels: int = 1, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:size, :size]
    blob = np.exp(-(((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / 20.0))
    rng = np.random.default_rng(1)
    shape = (1, channels, size, size)
    data = (blob * 1e-3 + rng.normal(0, 1e-5, (channels, size, size))).reshape(shape)
    header = fits.Header()
    header.update(BUNIT="Jy/beam", CTYPE3="FREQ", CRVAL3=100e9, CDELT3=1e9, CRPIX3=1,
                  BMAJ=2.5 / 3600, BMIN=2.0 / 3600)  # fmt: skip
    fits.PrimaryHDU(data.astype(np.float32), header).writeto(path, overwrite=True)


METADATA_COLUMNS = ["proposal_id", "ALMA_source_name", "Band", "Array_type", "Ang.res.",
                    "Obs.date", "member_ous_uid", "Freq"]  # fmt: skip


def metadata(argv: list[str]) -> None:
    say("querying the ALMA archive (fake)", "3 rows")
    out = Path(option(argv, "--save-csv") or "metadata.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(METADATA_COLUMNS)
        for i, source in enumerate(["Cloverleaf", "SDP9", "M82"]):
            w.writerow(["2023.1.00133.S", source, 4 + i, "12m", 1.5 + i / 10, "2025-03-18",
                        f"uid://A001/X1/X{i}", 126.0 + i])  # fmt: skip


def resolve(argv: list[str]) -> None:
    uids = (option(argv, "--member-ous-uid") or "").split(",")
    out = Path(option(argv, "--save-products-csv") or "products.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["access_url", "uid", "filename", "content_length", "content_type",
               "product_type", "semantics", "size_mb"]  # fmt: skip
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for uid in uids:
            for kind, size in (
                ("raw", 1_000_000_000),
                ("cubes", 200_000_000),
                ("auxiliary", 10_000),
            ):
                name = f"{uid.replace('/', '_')}_{kind}.tar"
                w.writerow([f"https://example/{name}", uid, name, size, "application/x-tar",
                            kind, "#this", size / 1e6])  # fmt: skip
    say(f"resolved {3 * len(uids)} products for {len(uids)} observations")


def download(argv: list[str]) -> None:
    dest = Path(option(argv, "--destination") or "downloads")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "download_manifest.json").write_text(
        json.dumps({"filter": option(argv, "--product-filter")})
    )
    say("downloading (fake)", "done")


def compute_parameters(argv: list[str]) -> None:
    folder, out = (positional(argv, 2) + [".", "imaging_parameters.csv"])[:2]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    columns = ["filename", "spectral_window_id", "reference_frequency", "fov_per_frequency",
               "max_baseline_size", "synthetized_beam_size", "n_channels", "duration_s",
               "target_field_ids", "field_name"]  # fmt: skip
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for spw, ghz in ((25, 95.6), (27, 97.4)):
            w.writerow([f"{folder}/uid___A002_X1_Xb670.ms.split.cal", spw, ghz * 1e9, 62.0,
                        250.0, 2.5, 240, 3170.0, "2", "H-ATLAS J113526"])  # fmt: skip
    say(f"2 windows in {folder}")


def image_from_ms(argv: list[str]) -> None:
    params, out = positional(argv, 2)[:2]
    with open(params, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        ms = Path(out) / Path(row["filename"]).stem
        task = ms / f"SPW-{row['spectral_window_id']}_FIELD-{row['target_field_ids']}"
        (ms / f"{task.name}.log").parent.mkdir(parents=True, exist_ok=True)
        (ms / f"{task.name}.log").write_text("wsclean (fake)\n")
        write_fits(task / "wsclean-image.fits")
        write_fits(task / "wsclean-residual.fits")
        say(f"{task.name}: imaged", pause=1.0)
        (ms / f"{task.name}.done").write_text("")


def clean_quality(argv: list[str]) -> None:
    root = Path(positional(argv, 2)[0])
    tasks = sorted(p for p in root.glob("*/SPW-*") if p.is_dir())
    rows = []
    for i, task in enumerate(tasks):
        quality = {"task": str(task), "continuum": {"snr": 40.0 + i, "residual_share_1beam": 0.05,
                   "residual_peak_sigma": 1.8, "model_flux_jy": 0.0008}}  # fmt: skip
        (task / "quality.json").write_text(json.dumps(quality))
        rows.append([f"{task.parent.name}/{task.name}", 40.0 + i, 0.05, 1.8, 0.0008])
    with (root / "clean-quality.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "continuum.snr", "continuum.residual_share_1beam",
                    "continuum.residual_peak_sigma", "continuum.model_flux_jy"])  # fmt: skip
        w.writerows(rows)
    say(f"{len(rows)} tasks assessed")


def simulation(argv: list[str]) -> None:
    out = Path(option(argv, "--output-dir") or "outputs")
    write_fits(out / "clean-cube_0.fits", channels=8)
    write_fits(out / "dirty-cube_0.fits", channels=8)
    say("simulated 1 run (fake)")


def preview_integrate(argv: list[str]) -> None:
    _source, out = positional(argv, 2)[:2]
    write_fits(Path(out))
    say(f"{out}: integrated (fake)")


def preview_wiener(argv: list[str]) -> None:
    out = Path(positional(argv, 2)[2])
    for name in ("model", "restored", "residual"):
        write_fits(out / f"wiener-{name}.fits", channels=8)
    say("wiener (fake)")


CANDIDATE_SHA = "0123456789ab" + "0" * 52
PACKAGE_MODEL = (
    Path(__file__).resolve().parents[2] / "src/almasim/services/noise_model/models/alma-1.json"
)


def noise_windows(argv: list[str]) -> None:
    out = Path(option(argv, "--out") or ".")
    out.mkdir(parents=True, exist_ok=True)
    names = ["uid___A002_X1_Xb670_spw25_field2", "uid___A002_X1_Xb670_spw27_field2"]
    wanted = option(argv, "--names")
    if wanted:
        names = [n for n in names if n in wanted.split(",")]
    else:
        say("  left out uid___A002_X1_Xb670_spw29_field0: field 0 is not a science target")
    with (out / "windows.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ms", "spw", "field", "name", "pwv", "task_log", "task_dir"])
        for name in names:
            spw = name.split("_spw")[1].split("_")[0]
            w.writerow(
                ["/data/cal/uid___A002_X1_Xb670.ms.split.cal", spw, "2", name, "0.9", "", ""]
            )
    with (out / "rows.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([*METADATA_COLUMNS, "name"])
        for name in names:
            w.writerow(["2023.1.00133.S", "H-ATLAS J113526", 3, "12m", 1.5, "2025-03-18",
                        "uid://A001/X1/X0", 95.6, name])  # fmt: skip
    say(f"{len(names)} window(s) -> {out / 'windows.csv'}")


def noise_step(argv: list[str]) -> None:
    out = Path(option(argv, "--out") or ".")
    step = argv[1]
    failures = (
        {"uid___A002_X1_Xb670_spw27_field2": "ValueError: too faint"} if step == "fit" else {}
    )
    name = "failures.json" if step == "extract" else f"failures-{step}.json"
    (out / name).write_text(json.dumps(failures))
    say(f"{step} (fake)", pause=0.5)
    if step == "validate":
        comparison = {
            "schema_version": 1,
            "model": {
                "name": "alma-candidate",
                "version": 2,
                "label": "alma-candidate@2",
                "sha256": CANDIDATE_SHA,
            },  # fmt: skip
            "windows": 2,
            "passed": True,
            "metrics": {
                "cont.sigma": {
                    "kind": "ratio",
                    "n": 2,
                    "median": 1.02,
                    "q16": 0.97,
                    "q84": 1.06,
                    "p": None,
                    "within_10pct": 2,
                }
            },  # fmt: skip
            "groups": {},
            "checks": [
                {
                    "name": "continuum noise",
                    "passed": True,
                    "value": 1.02,
                    "detail": "median 1.02 within 5%",
                }
            ],  # fmt: skip
        }
        folder = out / "validation" / "0123456789ab"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "comparison.json").write_text(json.dumps(comparison))


def noise_promote(argv: list[str]) -> None:
    """--site: the package's alma@1 as <name>@1, noted as the candidate's release."""
    profile = tomllib.loads(Path(os.environ["ALMASIM_SITE_PROFILE"]).read_text())
    name = option(argv, "--name") or "alma"
    model = json.loads(PACKAGE_MODEL.read_text())
    model.update(name=name, version=1)
    model["provenance"]["notes"] = f"(sha256 {CANDIDATE_SHA[:12]}) passed every check."
    path = Path(profile["tools"]["noise_models"]) / f"{name}-1.json"
    path.write_text(json.dumps(model))
    say(f"{name}@1 written to {path}")


def noise_aggregate(argv: list[str]) -> None:
    out = Path(option(argv, "--output") or "candidate.json")
    model = {
        "schema_version": 1,
        "name": "alma-candidate",
        "version": 2,
        "provenance": {
            "notes": "radiometer.receiver_temperature_k[9, 10]: 0 window(s), kept the base's",
            "values": {
                "channel_correlation.rho1": {"n": 2, "median": 0.667, "q16": 0.66, "q84": 0.67}
            },
        },
        "components": {},
    }
    out.write_text(json.dumps(model))
    say(f"2 window(s), 1 fitted -> alma-candidate@2 in {out}")


def doctor(argv: list[str]) -> None:
    say("ok    python     fake", "ok    slurm      fake")
    if "--slurm" in argv:  # long enough to cancel
        for i in range(60):
            say(f"waiting for the compute node ({i} s)", pause=1.0)


COMMANDS = {
    ("metadata",): metadata,
    ("products", "resolve"): resolve,
    ("products", "download"): download,
    ("image", "compute-parameters"): compute_parameters,
    ("image", "image-from-ms"): image_from_ms,
    ("image", "clean-quality"): clean_quality,
    ("simulation",): simulation,
    ("preview", "integrate"): preview_integrate,
    ("preview", "wiener"): preview_wiener,
    ("noise", "windows"): noise_windows,
    ("noise", "extract"): noise_step,
    ("noise", "fit"): noise_step,
    ("noise", "aggregate"): noise_aggregate,
    ("noise", "validate"): noise_step,
    ("noise", "promote"): noise_promote,
    ("doctor",): doctor,
}


def main(argv: list[str]) -> int:
    say("fake almasim " + " ".join(argv))
    for words, run in COMMANDS.items():
        if tuple(argv[: len(words)]) == words:
            run(argv)
            return 0
    print(f"fake almasim does not know {argv[:2]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
