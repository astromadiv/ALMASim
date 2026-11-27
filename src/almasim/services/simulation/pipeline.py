"""The whole simulation: clean cube, observation, images, export (MADIV-28 T5.7)."""

from __future__ import annotations

import time
from collections.abc import Callable
from time import gmtime, strftime

from .context import RunContext
from .export import export_results
from .params import LogFn, SimulationParams, StopFn
from .stages import generate_clean_cube, image_products, simulate_observation


def run_simulation(
    params: SimulationParams,
    *,
    logger: LogFn = None,
    status_callback: Callable[[str], None] | None = None,
    progress_emitter=None,
    compute_backend=None,
    terminal_logger=None,
    robust: float = 0.0,
    interferometer_progress_callback: Callable[[int], None] | None = None,
    stop_requested: StopFn = False,
):
    """Execute the full ALMA simulation workflow."""
    ctx = RunContext(logger, status_callback, stop_requested, remote=bool(params.remote))

    start = time.time()
    ctx.check_cancelled()
    ctx.status("Generating clean cube")
    clean_cube_stage = generate_clean_cube(
        params,
        logger=logger,
        status_callback=status_callback,
        progress_emitter=progress_emitter,
        compute_backend=compute_backend,
        terminal_logger=terminal_logger,
        stop_requested=ctx.stopped,
    )
    ctx.check_cancelled()
    ctx.status("Running interferometric simulation")
    simulation_results = simulate_observation(
        clean_cube_stage,
        compute_backend=compute_backend,
        robust=robust,
        terminal_logger=terminal_logger,
        interferometer_progress_callback=interferometer_progress_callback,
        image_products=not params.visibilities_only,
        apply_primary_beam=not params.apparent_input,
    )
    ctx.check_cancelled()
    ctx.status("Reconstructing image products")
    simulation_results = image_products(simulation_results)
    ctx.check_cancelled()
    exported_results = export_results(
        params,
        clean_cube_stage,
        simulation_results,
        logger=logger,
        status_callback=status_callback,
    )
    ctx.check_cancelled()

    ctx.log("Finished")
    stop = time.time()
    ctx.log("Execution took {} seconds".format(strftime("%H:%M:%S", gmtime(stop - start))))
    return exported_results
