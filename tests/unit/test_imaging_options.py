"""Imaging, self-calibration and execution options (MADIV-28 T4.1)."""

import pickle

import pytest

from almasim.services.imaging.options import ExecOptions, ImagingOptions, SelfcalOptions
from almasim.services.imaging.wsclean_plan import CubeOptions

pytestmark = pytest.mark.unit


def test_options_are_frozen_and_picklable():
    options = ImagingOptions(
        cube=CubeOptions(), cube_products=("image",), selfcal=SelfcalOptions(("inf",))
    )
    assert pickle.loads(pickle.dumps(options)) == options
    with pytest.raises(AttributeError):
        options.pbcor = True  # type: ignore[misc]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (ImagingOptions(cube=CubeOptions(), cube_products=("residual",)), "must include image"),
        (ImagingOptions(cube=CubeOptions(), cube_products=("image", "x")), "unknown: x"),
        (ImagingOptions(contsub=True), "need --cube"),
        (
            ImagingOptions(
                cube=CubeOptions(), cube_products=("image",), contsub=True, uv_contsub=True
            ),
            "drop --contsub",
        ),
        (ImagingOptions(selfcal=SelfcalOptions(("bogus",))), "--selfcal"),
        (
            ImagingOptions(single_window=False, selfcal=SelfcalOptions(("inf",))),
            "--single-window-ms",
        ),
        (
            ImagingOptions(
                cube=CubeOptions(), cube_products=("image",), uv_contsub=True, single_window=False
            ),
            "edits the copy",
        ),
        (ImagingOptions(continuum=False), "nothing to image"),
    ],
)
def test_validate_refuses_impossible_combinations(options, message):
    with pytest.raises(ValueError, match=message):
        options.validate()


def test_task_kwargs_apply_the_switches():
    options = ImagingOptions(
        cube=CubeOptions(),
        cube_products=("image",),
        uv_contsub=True,
        contsub_order=2,
        continuum_restart=True,
        cube_line_mask=True,
        selfcal=SelfcalOptions(("inf", "30s"), min_snr=10, noise_products=True),
    )
    options.validate()
    kwargs = options.task_kwargs("/casa")
    assert kwargs["uv_contsub_order"] == 2 and kwargs["contsub_order"] is None
    assert kwargs["selfcal_solints"] == ["inf", "30s"] and kwargs["casa_data"] == "/casa"
    assert kwargs["continuum_restart"] and kwargs["cube_line_mask"] and kwargs["noise_products"]
    plain = ImagingOptions(single_window=False, continuum_restart=True).task_kwargs()
    assert plain["selfcal_solints"] is None and plain["continuum_restart"] is False
    assert plain["pipeline_line_free"] is False and plain["ous_index"] is None


def test_pipeline_line_free_needs_a_line_free_selection():
    with pytest.raises(ValueError, match="--pipeline-line-free needs --uv-contsub or --selfcal"):
        ImagingOptions(pipeline_line_free=True).validate()
    options = ImagingOptions(
        cube=CubeOptions(),
        cube_products=("image",),
        uv_contsub=True,
        pipeline_line_free=True,
        ous_index="/tree/ous_index.csv",
    )
    options.validate()
    kwargs = options.task_kwargs()
    assert kwargs["pipeline_line_free"] and kwargs["ous_index"] == "/tree/ous_index.csv"
    ImagingOptions(pipeline_line_free=True, selfcal=SelfcalOptions(("inf",))).validate()


def test_exec_options_validate():
    ExecOptions().validate()
    with pytest.raises(ValueError, match="sync, slurm"):
        ExecOptions(backend="k8s").validate()
    with pytest.raises(ValueError, match="at least 1"):
        ExecOptions(n_jobs=0).validate()
