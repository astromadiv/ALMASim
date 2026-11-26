"""What every task of an imaging run makes, and where the tasks run (MADIV-28 T4.1).

Frozen and picklable: the driver builds them once from the command line,
``validate()`` checks the combination, and the tasks receive
:meth:`ImagingOptions.task_kwargs` on the workers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from almasim.services.imaging.wsclean_plan import CUBE_PRODUCTS, CubeOptions


@dataclass(frozen=True)
class SelfcalOptions:
    """Self-calibration of each task's continuum, round by round (none without ``solints``)."""

    solints: tuple[str, ...] = ()
    min_snr: float = 20.0
    # Keep the gain tables and write the correction and total-residual images.
    noise_products: bool = False

    def validate(self, *, continuum: bool, single_window: bool) -> None:
        from almasim.services.imaging.selfcal.loop import parse_selfcal_rounds

        try:
            parse_selfcal_rounds(self.solints)
        except ValueError as exc:
            raise ValueError(f"--selfcal: {exc}") from exc
        if self.solints and not (continuum and single_window):
            raise ValueError("--selfcal needs --continuum and --single-window-ms")


@dataclass(frozen=True)
class ImagingOptions:
    """What each task of a run makes."""

    continuum: bool = True
    cube: CubeOptions | None = None
    cube_products: tuple[str, ...] | None = None
    single_window: bool = True
    max_weight: float | None = None
    contsub: bool = False
    uv_contsub: bool = False
    contsub_order: int = 1
    # Also leave the ALMA pipeline's cont.dat line channels out of the
    # line-free channels; its member from ``ous_index`` (None: the nearest).
    pipeline_line_free: bool = False
    ous_index: str | None = None
    join_channels: bool = False
    continuum_restart: bool = False
    cube_line_mask: bool = False
    pbcor: bool = False
    # Also image all the windows of each field together (aggregate.py).
    aggregate_continuum: bool = False
    # Make that image from each window's continuum fitted over its line-free
    # channels (order ``contsub_order``), so the lines stay out of it.
    aggregate_line_free: bool = True
    selfcal: SelfcalOptions = field(default_factory=SelfcalOptions)

    def validate(self) -> None:
        """Refuse combinations a run cannot honour, with the option names in the message."""
        if self.cube is not None:
            products = self.cube_products or ()
            unknown = sorted(set(products) - set(CUBE_PRODUCTS))
            if "image" not in products or unknown:
                raise ValueError(
                    f"--cube-products must include image and only name {', '.join(CUBE_PRODUCTS)}"
                    + (f" (unknown: {', '.join(unknown)})" if unknown else "")
                )
        elif self.contsub or self.join_channels or self.uv_contsub:
            raise ValueError(
                "Options that need --cube: --plan-only, --contsub, --uv-contsub, --join-channels"
            )
        if self.uv_contsub and self.contsub:
            raise ValueError("--uv-contsub already removes the continuum: drop --contsub")
        self.selfcal.validate(continuum=self.continuum, single_window=self.single_window)
        fits_continuum = self.uv_contsub or (self.aggregate_continuum and self.aggregate_line_free)
        if self.pipeline_line_free and not (fits_continuum or self.selfcal.solints):
            raise ValueError(
                "--pipeline-line-free needs --uv-contsub, --selfcal or --aggregate-continuum"
            )
        if self.uv_contsub and not self.single_window:
            raise ValueError("--uv-contsub needs --single-window-ms: it edits the copy")
        if self.aggregate_continuum and not (self.continuum and self.single_window):
            raise ValueError("--aggregate-continuum needs --continuum and --single-window-ms")
        if not self.continuum and self.cube is None:
            raise ValueError("--no-continuum needs --cube: there would be nothing to image")

    def task_kwargs(self, casa_data: str | None = None) -> dict[str, Any]:
        """``run_wsclean_task`` keyword arguments shared by every task."""
        return {
            "single_window": self.single_window,
            "max_weight": self.max_weight,
            "cube_products": self.cube_products,
            "contsub_order": self.contsub_order if self.contsub else None,
            "uv_contsub_order": self.contsub_order if self.uv_contsub else None,
            "pipeline_line_free": self.pipeline_line_free,
            "ous_index": self.ous_index,
            "selfcal_solints": list(self.selfcal.solints) or None,
            "selfcal_min_snr": self.selfcal.min_snr,
            "casa_data": casa_data,
            "continuum_restart": self.continuum_restart and self.single_window,
            "cube_line_mask": (
                self.cube_line_mask and self.cube is not None and self.single_window
            ),
            "noise_products": self.selfcal.noise_products,
        }


@dataclass(frozen=True)
class ExecOptions:
    """Where and how a run's tasks execute."""

    backend: str = "slurm"
    cores_per_task: int = 10
    node_cores: int = 95
    queue: str = "normal"
    project: str | None = None
    walltime: str = "02:00:00"
    memory: str = "16GB"
    n_jobs: int = 1
    scheduler_host: str | None = None
    scheduler_interface: str | None = None
    task_timeout: float | None = None
    task_retries: int = 3
    continue_on_error: bool = True
    scratch_root: str | None = None

    def validate(self) -> None:
        if self.backend not in ("sync", "slurm"):
            raise ValueError("--backend must be one of: sync, slurm.")
        if self.cores_per_task < 1 or self.n_jobs < 1:
            raise ValueError("--num-cores and --slurm-workers must be at least 1")
