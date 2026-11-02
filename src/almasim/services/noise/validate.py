"""Simulate real windows with a noise model and measure them as the real ones
were (MADIV-28): what ``noise validate`` does per window.

For each window: ALMASim simulates it from its archive row (a point source as
bright as the real image's continuum peak, the window's frequency, channels
and PWV, one execution block, the model under test), exports the MS, flags
as large a share of its rows as the real image is missing (the window's
flagged data and the rows imaging leaves out for their WEIGHT), measures it
with ``noise extract`` and images it as the real window was
imaged (``image compute-parameters``, ``image image-from-ms`` with the cube
and continuum subtraction, ``image clean-quality``), then takes the image
statistics of both. Everything goes under
``<out>/validation/<model sha256[:12]>/windows/<name>/``, so the results of
different models never mix; :mod:`.compare` judges them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..noise_model import NoiseModel
from .batch import WINDOW_FILE, WindowTask, _write_atomic
from .compare import Comparison, Criteria, Pair, compare
from .image_stats import image_noise_stats
from .measurements import WindowNoise

SIM_WINDOW = "sim_window.json"
SIM_IMAGE = "sim_image.json"
REAL_IMAGE = "real_image.json"
MAX_CHANNELS = 960
MAX_UNIMAGED = 0.95  # the most of a simulated window's data flag_as_imaged flags
FLAG_CHUNK_ROWS = 10_000


@dataclass(frozen=True)
class Imaging:
    """How the simulated windows are imaged (``image image-from-ms`` options)."""

    wsclean_bin: str | None = None
    casa_data_root: str | None = None
    num_cores: int = 8
    task_memory_gb: float = 40.0
    task_timeout: float = 10800.0
    extra: tuple[str, ...] = field(default_factory=tuple)

    def arguments(self) -> list[str]:
        args = [
            "--backend", "sync", "--cube", "--uv-contsub",
            "--num-cores", str(self.num_cores),
            "--task-memory-gb", str(self.task_memory_gb),
            "--task-timeout", str(int(self.task_timeout)),
        ]  # fmt: skip
        if self.wsclean_bin:
            args += ["--wsclean-bin", self.wsclean_bin]
        if self.casa_data_root:
            args += ["--casa-data-root", self.casa_data_root]
        return args + list(self.extra)


def validation_dir(out: Path, model: NoiseModel) -> Path:
    return out / "validation" / model.sha256()[:12]


def done(task: WindowTask, out: Path, model: NoiseModel) -> bool:
    directory = validation_dir(out, model) / "windows" / task.name
    return all((directory / name).is_file() for name in (SIM_WINDOW, SIM_IMAGE, REAL_IMAGE))


def validate_window(
    task: WindowTask,
    row: pd.Series,
    model: NoiseModel,
    *,
    measured: Path,
    out: Path,
    imaging: Imaging,
    seed: int = 0,
    keep_ms: bool = False,
) -> Path:
    """Simulate, measure and image one window; its directory."""
    real = WindowNoise.model_validate_json(
        (measured / "windows" / task.name / WINDOW_FILE).read_text()
    )
    if not task.task_dir:
        raise ValueError(f"{task.name}: needs its real imaging task (task_dir) to compare with")
    directory = validation_dir(out, model) / "windows" / task.name
    directory.mkdir(parents=True, exist_ok=True)
    _write_atomic(validation_dir(out, model) / "model.json", model.model_dump_json(indent=2) + "\n")
    if not (directory / REAL_IMAGE).is_file():
        _write_atomic(
            directory / REAL_IMAGE, json.dumps(image_noise_stats(task.task_dir), indent=1) + "\n"
        )
    real_peak = json.loads((directory / REAL_IMAGE).read_text()).get("q.continuum.peak_jy_beam")
    run = directory / "sim"
    if run.exists():
        shutil.rmtree(run)
    ms = simulate(
        row,
        real,
        model,
        run,
        seed=seed + zlib.crc32(task.name.encode()) % 1_000_003,
        peak_jy=real_peak,
    )
    flag_as_imaged(ms, real, seed=seed + zlib.crc32(task.name.encode()) % 1_000_003)
    sim_window = measure(ms, task.name, run)
    _write_atomic(directory / SIM_WINDOW, sim_window.model_dump_json(indent=2) + "\n")
    task_dir = image(run, imaging)
    _write_atomic(directory / SIM_IMAGE, json.dumps(image_noise_stats(task_dir), indent=1) + "\n")
    if not keep_ms:
        shutil.rmtree(ms, ignore_errors=True)
    return directory


def run_in_process(
    todo: Sequence[WindowTask],
    *,
    rows: dict[str, Any],
    model: NoiseModel,
    measured: Path,
    out: Path,
    imaging: Imaging,
    seed: int = 0,
    keep_ms: bool = False,
    echo: Callable[[str], Any] = print,
) -> dict[str, str | None]:
    """Validate the windows here, one after the other."""
    results: dict[str, str | None] = {}
    for task in todo:
        try:
            path = validate_window(
                task,
                rows[task.name],
                model,
                measured=measured,
                out=out,
                imaging=imaging,
                seed=seed,
                keep_ms=keep_ms,
            )
            echo(f"{task.name}: {path}")
            results[task.name] = None
        except Exception as exc:  # noqa: BLE001 - one window must not stop the others
            echo(f"{task.name}: FAILED {exc}")
            results[task.name] = f"{type(exc).__name__}: {exc}"
    return results


def simulate(
    row: pd.Series,
    window: WindowNoise,
    model: NoiseModel,
    run: Path,
    *,
    seed: int,
    peak_jy: float | None = None,
) -> Path:
    """ALMASim's simulation of the window: its archive row, moved to the
    window's frequency and channels and observed as long, by as many
    antennas and through as much water vapour as the window's MS, a point
    source (``peak_jy`` of continuum, when given), one execution block, the
    model; the exported MS."""
    import almasim
    from almasim.services import astro
    from almasim.services.astro.spectral import with_redshifts
    from almasim.services.compute import create_backend
    from almasim.services.simulation import (
        SimulationParams,
        export_results,
        generate_clean_cube,
        simulate_observation,
    )

    main_dir = Path(almasim.__file__).parent
    record = row.copy()
    record["Freq"] = window.exposure.frequency_ghz
    if support := frequency_support(window):
        record["Freq.sup."] = support
    record = with_exposure(record, window, seed)
    record = with_weather(record, window, model)
    rest_frequency, _ = astro.get_line_info(main_dir)
    record = with_redshifts(pd.DataFrame([record]), rest_frequency).iloc[0]
    run.mkdir(parents=True, exist_ok=True)
    params = SimulationParams.from_metadata_row(
        record,
        idx=0,
        main_dir=main_dir,
        output_dir=run,
        tng_dir=run / "tng",
        galaxy_zoo_dir=run / "galaxy_zoo",
        hubble_dir=run / "hubble",
        project_name="validate",
        source_type="point",
        save_mode="fits",
        persist=True,
        ml_dataset_path=None,
        n_pix=256,
        n_channels=int(min(window.exposure.channels, MAX_CHANNELS)),
        seed=seed,
        noise_model=model,
        executions=1,
        ms_export=True,
        ms_save_mode="msv2",
    )
    with create_backend("sync") as backend:
        clean = generate_clean_cube(params, logger=print, compute_backend=backend)
        if peak_jy is not None:
            clean = with_continuum(clean, peak_jy)
        results = simulate_observation(clean, compute_backend=backend)
        exported = export_results(params, clean, results, logger=print)
    return Path(exported["ms_path"])


def with_continuum(clean: Any, peak_jy: float) -> Any:
    """The clean-cube stage with its source scaled to ``peak_jy`` of continuum
    (its continuum's mean flux over the channels), the real image's continuum
    peak. The archive row's sensitivities make the source some 100-1000 times
    the continuum's noise; its cleaning residuals then dominate the simulated
    continuum's noise where the real field is faint. Unchanged without a
    positive peak or a source."""
    from dataclasses import replace

    import numpy as np

    flux = np.sum(clean.model_cube, axis=(1, 2))
    if clean.continuum_fraction is not None:
        flux = flux * clean.continuum_fraction
    level = float(np.mean(flux))
    if not (peak_jy > 0 and level > 0):
        return clean
    model = clean.model_cube * (peak_jy / level)
    runs = [replace(run, skymodel=model) for run in clean.interferometer_runs]
    return replace(clean, model_cube=model, interferometer_runs=runs)


def frequency_support(window: WindowNoise) -> str | None:
    """The window's own entry in the archive's frequency support (its band and
    resolution, from its MS), so that it is simulated with its own channel
    averaging: the archive row's support lists the windows of the whole
    observing unit, whose resolution may not be this window's. None when the
    window's measurement has no resolution."""
    e = window.exposure
    if not e.resolution_hz:
        return None
    half = e.channels * e.channel_width_hz / 2e9
    low, high = e.frequency_ghz - half, e.frequency_ghz + half
    return f"[{low:.9f}..{high:.9f}GHz,{e.resolution_hz / 1e3:.6f}kHz]"


def with_exposure(record: pd.Series, window: WindowNoise, seed: int) -> pd.Series:
    """The archive row observed as the window's MS was: its time on source in
    one execution block, by the MS's own antennas on their pads
    (``exposure.pads``). The row's Int.Time is the whole observing unit's,
    which may be several of the MS's blocks, and its antennas are the union
    over them, configurations and the total-power antennas included. A window
    measured without its pads takes as many of the row's antennas as the MS
    has, one per pad, drawn with ``seed``."""
    import numpy as np

    e = window.exposure
    record = record.copy()
    record["Int.Time"] = e.time_on_source_s
    record["n_executions"] = 1
    if e.pads:
        record["antenna_arrays"] = " ".join(e.pads)
        return record
    pads: dict[str, str] = {}
    for token in str(record.get("antenna_arrays", "")).split():
        pads.setdefault(token.split(":")[0], token)
    tokens = list(pads.values())
    if 2 <= e.antennas < len(tokens):
        picked = np.random.default_rng(seed).choice(len(tokens), size=e.antennas, replace=False)
        tokens = [tokens[i] for i in np.sort(picked)]
    if tokens:
        record["antenna_arrays"] = " ".join(tokens)
    return record


def with_weather(record: pd.Series, window: WindowNoise, model: NoiseModel) -> pd.Series:
    """The archive row with the window's PWV: its own (its execution block's,
    noise extract --pwv), or without one the PWV at which the model's Tsys is
    ALMA's (pwv_from_tsys). The row's is the observing unit's, one of its
    blocks', and a 7 m array's is the PWV it was scheduled for. Unchanged when
    there is neither."""
    pwv = window.input.pwv_mm
    if pwv is None:
        pwv = pwv_from_tsys(window, model)
    if pwv is None:
        return record
    record = record.copy()
    record["PWV"] = pwv
    return record


def pwv_from_tsys(window: WindowNoise, model: NoiseModel) -> float | None:
    """The PWV (mm, within 0-10) at which the model's Tsys (its receiver
    temperatures and equation) at the window's frequencies and elevation is the
    window's median SYSCAL Tsys; None without a Tsys or an elevation. The Tsys
    then follows ALMA's by construction, so the window still tests the rest:
    the efficiencies, the noise's shape and everything after it."""
    import numpy as np
    from scipy.optimize import brentq

    from ..interferometry import noise
    from ..interferometry.atmosphere import system_temperature

    e = window.exposure
    if window.tsys is None or e.elevation_deg is None:
        return None
    config = noise.NoiseModelConfig(pwv_mm=1.0, **model.noise_config())
    freq = e.frequency_ghz * 1e9 + np.linspace(-0.5, 0.5, 64) * e.channels * e.channel_width_hz
    trx = noise.estimate_receiver_temperature_k(
        freq, config.receiver_model, config.receiver_temperatures
    )
    airmass = 1.0 / np.sin(np.radians(max(5.0, min(90.0, e.elevation_deg))))
    target = window.tsys.median_k

    def excess(pwv: float) -> float:
        tsys = system_temperature(
            freq,
            trx,
            pwv_mm=pwv,
            airmass=airmass,
            ground_temperature_k=config.ground_temperature_k,
            forward_efficiency=config.forward_efficiency,
        )
        return float(np.median(tsys)) - target

    low, high = 0.0, 10.0
    if excess(low) >= 0:
        return low
    if excess(high) <= 0:
        return high
    return float(brentq(excess, low, high, xtol=1e-3))


def unimaged_fraction(window: WindowNoise) -> float:
    """The share of the real window's data its image does without: its flagged
    data and its rows whose WEIGHT exceeds the imaging's maximum (rows the
    archive pipeline flagged and calibration left unflagged, so the two are
    apart). At most ``MAX_UNIMAGED``."""
    e = window.exposure
    return float(min(max(e.flagged_fraction + e.high_weight_fraction, 0.0), MAX_UNIMAGED))


def flag_as_imaged(ms: Path, window: WindowNoise, *, seed: int) -> int:
    """Flag rows of the simulated MS, drawn with ``seed``, until it misses as
    large a share of its data as the real window's image (``unimaged_fraction``);
    the rows newly flagged. In a process of its own, as ``measure`` is."""
    from ..runtime.child import call_in_child
    from ..runtime.env import wsclean_environment

    fraction = unimaged_fraction(window)
    if fraction <= 0:
        return 0
    result = call_in_child(
        "almasim.services.noise.validate:flag_rows",
        {"ms_path": str(ms), "fraction": fraction, "seed": int(seed)},
        what="flagging the simulated MS",
        env=wsclean_environment(1),
    )
    return int(result["flagged_rows"])


def flag_rows(ms_path: str, fraction: float, seed: int) -> dict[str, int]:
    """Flag whole cross-correlation rows of ``ms_path``, drawn with ``seed``
    from those with no flag, until ``fraction`` of its cross-correlation data
    is flagged (counting what already is)."""
    import numpy as np

    from ..imaging.ms_geometry import import_casacore_tables

    table = import_casacore_tables()
    with table(ms_path, readonly=False, ack=False) as main:
        nrow = main.nrows()
        cross = main.getcol("ANTENNA1") != main.getcol("ANTENNA2")
        share = np.zeros(nrow)
        for start in range(0, nrow, FLAG_CHUNK_ROWS):
            flag = main.getcol("FLAG", start, FLAG_CHUNK_ROWS)
            share[start : start + len(flag)] = flag.mean(axis=(1, 2))
        already = float(share[cross].sum())
        free = np.flatnonzero(cross & (share == 0))
        count = int(np.clip(round(fraction * cross.sum() - already), 0, free.size))
        picked = np.zeros(nrow, dtype=bool)
        picked[np.random.default_rng(seed).choice(free, size=count, replace=False)] = True
        for start in range(0, nrow, FLAG_CHUNK_ROWS):
            chunk = picked[start : start + FLAG_CHUNK_ROWS]
            if chunk.any():
                flag = main.getcol("FLAG", start, chunk.size)
                flag[chunk] = True
                main.putcol("FLAG", flag, start, chunk.size)
                flag_row = main.getcol("FLAG_ROW", start, chunk.size)
                main.putcol("FLAG_ROW", flag_row | chunk, start, chunk.size)
    return {"flagged_rows": count}


def measure(ms: Path, name: str, run: Path) -> WindowNoise:
    """``noise extract`` of the simulated MS, in a process of its own: the
    simulation wrote it with casatools, and python-casacore opening it in the
    same process can abort (free(): invalid pointer)."""
    from ...jobs.wrapper import ALMASIM_COMMAND

    out = run / "measured"
    command = ["noise", "extract", str(ms), "--spw", "0", "--name", name, "--out", str(out)]
    result = subprocess.run([*ALMASIM_COMMAND, *command], capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip()[-1500:]
        raise RuntimeError(f"almasim noise extract failed: {tail}")
    return WindowNoise.model_validate_json((out / "windows" / name / WINDOW_FILE).read_text())


def image(run: Path, imaging: Imaging) -> Path:
    """Image the run's MS as the real windows were; the imaging task directory."""
    from ...jobs.wrapper import ALMASIM_COMMAND

    params, cubes = run / "imaging-params.csv", run / "cubes"
    for command in (
        ["image", "compute-parameters", str(run), str(params), "--pattern", "*.ms"],
        ["image", "image-from-ms", str(params), str(cubes), *imaging.arguments()],
        ["image", "clean-quality", str(cubes)],
    ):
        result = subprocess.run([*ALMASIM_COMMAND, *command], capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip()[-1500:]
            raise RuntimeError(f"almasim {' '.join(command[:2])} failed: {tail}")
    tasks = sorted(cubes.glob("*/SPW-0_FIELD-0"))
    if not tasks:
        raise RuntimeError(f"no imaging task under {cubes}")
    return tasks[0]


def collect(
    tasks: Sequence[WindowTask], out: Path, model: NoiseModel, measured: Path
) -> list[Pair]:
    """The windows validated for ``model``, as pairs to compare."""
    from ..interferometry.atmosphere import receiver_band

    pairs = []
    for task in tasks:
        directory = validation_dir(out, model) / "windows" / task.name
        if not all((directory / n).is_file() for n in (SIM_WINDOW, SIM_IMAGE, REAL_IMAGE)):
            continue
        real_path = measured / "windows" / task.name / WINDOW_FILE
        real = (
            WindowNoise.model_validate_json(real_path.read_text()) if real_path.is_file() else None
        )
        band = ""
        if real is not None:
            number = receiver_band(real.exposure.frequency_ghz * 1e9).name
            band = f"band {number} {real.exposure.dish_diameter_m:g} m"
        pairs.append(
            Pair(
                name=task.name,
                sim_image=json.loads((directory / SIM_IMAGE).read_text()),
                real_image=json.loads((directory / REAL_IMAGE).read_text()),
                sim_window=WindowNoise.model_validate_json((directory / SIM_WINDOW).read_text()),
                real_window=real,
                band=band,
            )
        )
    return pairs


def judge(
    tasks: Sequence[WindowTask],
    out: Path,
    model: NoiseModel,
    measured: Path,
    criteria: Criteria | None = None,
) -> Comparison:
    """Compare the validated windows and write comparison.json."""
    comparison = compare(collect(tasks, out, model, measured), model.identity(), criteria)
    path = validation_dir(out, model) / "comparison.json"
    _write_atomic(path, comparison.model_dump_json(indent=2) + "\n")
    return comparison


def metadata_rows(path: Path) -> dict[str, Any]:
    """Archive rows by window name (a ``name`` column)."""
    table = pd.read_csv(path)
    if "name" not in table.columns:
        raise ValueError(f"{path}: needs a name column naming each row's window")
    return {str(row["name"]): row for _, row in table.iterrows()}
