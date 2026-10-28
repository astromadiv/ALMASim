"""CLI example for the staged ALMASim simulation API.

This script demonstrates:
1. metadata query or CSV loading
2. clean cube generation
3. in-memory interferometric simulation
4. ML HDF5 shard export for DDRM-style training/validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from almasim import export_results, generate_clean_cube, simulate_observation
from almasim.services import astro
from almasim.services.astro.spectral import sample_given_redshift
from almasim.services.compute import create_backend
from almasim.services.metadata.tap import (
    InclusionFilters,
    load_metadata,
    query_metadata_by_science,
)
from almasim.services.simulation import SimulationParams


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-mode",
        type=str,
        default="query",
        choices=["query", "csv"],
        help="How to obtain metadata rows for the simulation.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional CSV file containing pre-fetched ALMA metadata rows.",
    )
    parser.add_argument(
        "--science-keyword",
        action="append",
        default=[],
        help="Science keyword filter. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--scientific-category",
        action="append",
        default=[],
        help="Scientific category filter. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--band",
        action="append",
        type=int,
        default=[],
        help="ALMA band filter. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help="Optional source-name substring filter for the TAP query.",
    )
    parser.add_argument(
        "--metadata-limit",
        type=int,
        default=25,
        help="Maximum number of metadata rows to keep after querying/loading.",
    )
    parser.add_argument(
        "--save-metadata-csv",
        type=Path,
        default=None,
        help="Optional path to save the queried/filtered metadata as CSV.",
    )
    parser.add_argument(
        "--row-idx",
        type=int,
        default=0,
        help="Row index from the metadata CSV to simulate.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="ddrm_demo",
        help="Project name used in generated outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "examples" / "output",
        help="Directory used for optional standard outputs and ML shards.",
    )
    parser.add_argument(
        "--main-dir",
        type=Path,
        default=repo_root / "src" / "almasim",
        help="Path to the ALMASim data directory.",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="point",
        choices=[
            "point",
            "gaussian",
            "diffuse",
            "galaxy-zoo",
            "molecular",
            "hubble-100",
        ],
        help="Sky model family for clean cube generation.",
    )
    parser.add_argument(
        "--n-pix",
        type=int,
        default=128,
        help="Override spatial cube size.",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=32,
        help="Override channel count.",
    )
    parser.add_argument(
        "--robust",
        type=float,
        default=0.0,
        help="Robust weighting parameter for imaging.",
    )
    parser.add_argument(
        "--save-mode",
        type=str,
        default="memory",
        choices=["memory", "npz", "h5", "fits"],
        help="Standard output mode. Use 'memory' for notebook/ML workflows.",
    )
    parser.add_argument(
        "--persist-standard-outputs",
        action="store_true",
        help="Persist standard ALMASim outputs in addition to the ML shard.",
    )
    parser.add_argument(
        "--ml-shard-path",
        type=Path,
        default=repo_root / "examples" / "output" / "ddrm_training_sample.h5",
        help="Destination HDF5 shard for ML training/validation.",
    )
    return parser


def load_or_query_metadata(args: argparse.Namespace) -> pd.DataFrame:
    if args.metadata_mode == "csv":
        if args.metadata_path is None:
            raise SystemExit("--metadata-path is required when --metadata-mode=csv")
        metadata = load_metadata(args.metadata_path)
    else:
        science_keyword = args.science_keyword or ["Galaxies"]
        band = args.band or [6]
        include = InclusionFilters(
            science_keyword=science_keyword,
            scientific_category=args.scientific_category or None,
            band=band,
            source_name=args.source_name,
        )
        metadata = query_metadata_by_science(include=include)

    if metadata.empty:
        raise SystemExit("No metadata rows matched the requested filters")

    if args.metadata_limit is not None:
        metadata = metadata.head(args.metadata_limit).reset_index(drop=True)

    if args.save_metadata_csv is not None:
        save_path = args.save_metadata_csv.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(save_path, index=False)
        print(f"Saved metadata CSV: {save_path}")

    return metadata


def build_params(args: argparse.Namespace, metadata: pd.DataFrame) -> SimulationParams:
    rest_frequency, _ = astro.get_line_info(args.main_dir)
    sampled = sample_given_redshift(
        metadata,
        n=max(args.row_idx + 1, 1),
        rest_frequency=rest_frequency,
        extended=(args.source_type == "extended"),
        zmax=None,
    )
    if args.row_idx >= len(sampled):
        raise SystemExit(
            f"--row-idx {args.row_idx} is out of range for {len(sampled)} metadata rows"
        )
    row = sampled.iloc[args.row_idx]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ml_shard_path = args.ml_shard_path.expanduser().resolve()

    return SimulationParams.from_metadata_row(
        row,
        idx=args.row_idx,
        main_dir=args.main_dir,
        output_dir=args.output_dir,
        tng_dir=args.output_dir / "tng",
        galaxy_zoo_dir=args.output_dir / "galaxy_zoo",
        hubble_dir=args.output_dir / "hubble",
        project_name=args.project_name,
        source_type=args.source_type,
        save_mode=args.save_mode,
        persist=args.persist_standard_outputs,
        ml_dataset_path=ml_shard_path,
        n_pix=args.n_pix,
        n_channels=args.n_channels,
        # No MeasurementSet: it needs casatools or python-casacore.
        ms_save_mode="none",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    metadata = load_or_query_metadata(args)
    print(f"Metadata rows available: {len(metadata)}")
    print(f"Using row index: {args.row_idx}")
    params = build_params(args, metadata)
    print(f"Selected source: {params.source_name} ({params.member_ouid})")

    with create_backend("sync") as backend:
        clean_stage = generate_clean_cube(
            params,
            logger=print,
            compute_backend=backend,
        )
        print(f"Clean cube shape: {clean_stage.model_cube.shape}")
        print(f"Target ML shard: {params.ml_dataset_path}")

        simulation_results = simulate_observation(
            clean_stage,
            compute_backend=backend,
            robust=args.robust,
        )
        print(f"Dirty cube shape: {simulation_results['dirty_cube'].shape}")
        print(f"UV mask cube shape: {simulation_results['uv_mask_cube'].shape}")

        exported_results = export_results(
            params,
            clean_stage,
            simulation_results,
            logger=print,
        )

    print("Run complete")
    print(f"ML shard written to: {exported_results.get('ml_dataset_path')}")
    print(
        "Exported keys: "
        + ", ".join(
            sorted(
                key
                for key in exported_results.keys()
                if key.endswith("_cube") or key.endswith("_path")
            )
        )
    )


if __name__ == "__main__":
    main()
