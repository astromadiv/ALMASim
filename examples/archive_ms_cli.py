"""CLI example for importing raw ALMA MSs and creating calibrated MSs.

This example runs both archive service stages:

1. ASDM unpack/import to raw MeasurementSet via ``archive.unpack``.
2. Delivered calibration application and science-SPW split via
   ``archive.calibration``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=("ALMA delivery root or member directory containing raw/ and calibration/ products."),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "examples" / "output" / "archive_ms",
        help="Root directory where raw_ms/ and calibrated_ms/ outputs are written.",
    )
    parser.add_argument(
        "--asdm",
        default=None,
        help=("Optional ASDM UID without .asdm.sdm suffix. If omitted, all ASDMs are processed."),
    )
    parser.add_argument(
        "--casa-data-root",
        type=Path,
        default=None,
        help="Optional populated CASA runtime data directory.",
    )
    parser.add_argument(
        "--skip-casa-data-update",
        action="store_true",
        help=("Do not download CASA runtime data if the selected data directory is empty."),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing raw/calibrated MS outputs.",
    )
    parser.add_argument(
        "--skip-unpack",
        action="store_true",
        help="Skip raw ASDM import and calibrate existing raw_ms output.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Only import raw ASDMs; do not create calibrated MSs.",
    )
    parser.add_argument(
        "--clean-intermediate",
        action="store_true",
        help=("After calibrated MSs are created, remove raw_ms and calibration working data."),
    )
    parser.add_argument(
        "--delete-input-root",
        action="store_true",
        help=(
            "With --clean-intermediate, also remove the input delivery tree "
            "if it does not contain calibrated output."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from almasim.services.archive import (
        create_calibrated_measurement_sets,
        create_measurement_sets,
        organize_delivery,
    )

    output_root = args.output_root.expanduser().resolve()
    raw_ms_root = output_root / "raw_ms"
    calibrated_ms_root = output_root / "calibrated_ms"

    print(f"Input root: {args.input_root.expanduser().resolve()}")
    print(f"Output root: {output_root}")
    print(f"Raw MS root: {raw_ms_root}")
    print(f"Calibrated MS root: {calibrated_ms_root}")

    if not args.skip_unpack:
        raw_mss = create_measurement_sets(
            input_root=args.input_root,
            output_root=raw_ms_root,
            asdm_uid=args.asdm,
            casa_data_root=args.casa_data_root,
            skip_casa_data_update=args.skip_casa_data_update,
            overwrite=args.overwrite,
            logger_fn=print,
        )
        print("Raw MS products:")
        for raw_ms in raw_mss:
            print(f"  {raw_ms}")
    else:
        print("Skipping raw ASDM import.")

    if args.skip_calibration:
        print("Skipping calibration.")
        return

    calibrated_mss = create_calibrated_measurement_sets(
        input_root=args.input_root,
        raw_ms_root=raw_ms_root,
        output_root=calibrated_ms_root,
        asdm_uid=args.asdm,
        casa_data_root=args.casa_data_root,
        skip_casa_data_update=args.skip_casa_data_update,
        overwrite=args.overwrite,
        clean_intermediate=args.clean_intermediate,
        original_data_root=args.input_root if args.delete_input_root else None,
        logger_fn=print,
    )
    print("Calibrated MS products:")
    for calibrated_ms in calibrated_mss:
        print(f"  {calibrated_ms}")

    # Link each calibrated MS into its member's calibrated/ folder and index the tree.
    if not args.delete_input_root:
        organize_delivery(args.input_root, calibrated_ms_root, raw_ms_root, echo=print)


if __name__ == "__main__":
    main()
