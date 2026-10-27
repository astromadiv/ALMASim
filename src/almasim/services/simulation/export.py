"""Write a run's products to disk (MADIV-28 T5.7)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from ..interferometry import antenna as ual_antenna
from ..ms.io import export_native_ms
from ..products.cube_export import save_optional_cube, write_ml_dataset_shard
from .context import RunContext
from .params import LogFn, SimulationParams
from .rng import run_rng
from .stages import CleanCubeStage


def export_results(  # noqa: C901
    params: SimulationParams,
    clean_cube_stage: CleanCubeStage,
    simulation_results: dict[str, Any],
    *,
    logger: LogFn = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Persist optional file exports for simulation results."""
    ctx = RunContext(logger, status_callback, remote=bool(params.remote))
    log, status = ctx.log, ctx.status

    exported_results = dict(simulation_results)
    exported_results["background_cube"] = clean_cube_stage.background_cube
    persist_outputs = bool(params.persist and params.save_mode != "memory")

    status("Exporting results")
    if persist_outputs:
        if clean_cube_stage.sim_output_dir is not None:
            antenna_config_paths = []
            cube_output_paths: dict[str, list[str]] = {}
            for index, observation_config in enumerate(
                clean_cube_stage.observation_plan["configs"]
            ):
                if len(clean_cube_stage.observation_plan["configs"]) == 1:
                    output_dir = clean_cube_stage.sim_output_dir
                    output_name = "antenna.cfg"
                else:
                    output_dir = os.path.join(
                        clean_cube_stage.sim_output_dir,
                        f"config_{index}",
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    output_name = "antenna.cfg"
                ual_antenna.generate_antenna_config_file_from_antenna_array(
                    observation_config["antenna_array"],
                    params.main_dir,
                    output_dir,
                )
                antenna_config_paths.append(os.path.join(output_dir, output_name))
            exported_results["antenna_config_path"] = antenna_config_paths[0]
            exported_results["antenna_config_paths"] = antenna_config_paths
            optional_cubes = (
                ("int_image_cube", "int-image-cube", exported_results.get("int_image_cube")),
                ("tp_image_cube", "tp-image-cube", exported_results.get("tp_image_cube")),
                (
                    "tp_int_image_cube",
                    "tp-int-image-cube",
                    exported_results.get("tp_int_image_cube"),
                ),
                ("tp_dirty_cube", "tp-dirty-cube", exported_results.get("tp_dirty_cube")),
                ("background_cube", "background-cube", clean_cube_stage.background_cube),
            )
            for key, product, cube in optional_cubes:
                cube_output_paths[f"{key}_paths"] = save_optional_cube(
                    clean_cube_stage.sim_output_dir,
                    product,
                    params.idx,
                    cube,
                    params.save_mode,
                    clean_cube_stage.header,
                )
            exported_results.update(
                {key: value for key, value in cube_output_paths.items() if value}
            )

        exported_results["sim_params_path"] = clean_cube_stage.sim_params_payload.write()
    else:
        log("Skipping on-disk simulation exports (pure Python mode)")

    visibility_table = exported_results.get("visibility_table")
    _ms_mode = (
        params.ms_save_mode
        if params.ms_save_mode != "none"
        else ("msv2" if params.ms_export else "none")
    )
    if _ms_mode != "none":
        if visibility_table is None:
            raise RuntimeError(
                "MS/visibility export requested, but no visibility table is available"
            )
        _out_dir = (
            clean_cube_stage.sim_output_dir
            if clean_cube_stage.sim_output_dir is not None
            else clean_cube_stage.output_dir_abs
        )
        _base = os.path.join(_out_dir, f"{params.project_name}_{params.idx}")
        if _ms_mode in ("npz", "both"):
            vis_npz_path = _base + "_vis.npz"
            arrays: dict[str, Any] = {
                k: np.asarray(v) if hasattr(v, "__len__") else v
                for k, v in visibility_table.items()
            }
            np.savez_compressed(vis_npz_path, **arrays)
            exported_results["vis_npz_path"] = vis_npz_path
        if _ms_mode in ("msv2", "both"):
            ms_path = params.ms_export_dir if params.ms_export_dir is not None else _base + ".ms"
            exported_results["ms_path"] = export_native_ms(
                ms_path=ms_path,
                visibility_table=visibility_table,
                project_name=params.project_name,
                source_name=params.source_name,
                rng=run_rng(params.seed, "polarization"),
                history=(
                    (f"noise_model={json.dumps(params.noise_model.identity())}",)
                    if params.noise_model is not None
                    else ()
                ),
            )

    if params.ml_dataset_path:
        ml_dataset_path = write_ml_dataset_shard(
            params.ml_dataset_path,
            clean_cube=exported_results["model_cube"],
            dirty_cube=exported_results["dirty_cube"],
            dirty_vis=exported_results["dirty_vis"],
            uv_mask_cube=exported_results["uv_mask_cube"],
            metadata=clean_cube_stage.metadata,
        )
        exported_results["ml_dataset_path"] = ml_dataset_path

    return exported_results
