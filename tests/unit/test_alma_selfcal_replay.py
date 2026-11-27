"""Replay of the ALMA pipeline's self-calibration from its CASA commands log (MADIV-28)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from almasim.services.imaging.selfcal import alma_replay as replay
from almasim.services.imaging.wsclean_plan import member_selfcal

EB1, EB2 = "uid___A002_X1_X2", "uid___A002_X3_X4"

# The shape of a member's casa_commands.log around hif_selfcal (AS 209,
# uid://A001/X3621/X2b6c), its lines wrapped at 80 columns as the pipeline's are.
LOG = f"""# hif_mstransform()
#
# No comment registered for hif_mstransform
#
mstransform(vis='{EB1}.ms',
            outputvis='{EB1}_targets.ms', field='AS_209',
            spw='25,27', intent='OBSERVE_TARGET#ON_SOURCE',
            datacolumn='corrected', reindex=False)
mstransform(vis='{EB2}.ms', outputvis='{EB2}_targets.ms', field='AS_209',
            spw='25,27', datacolumn='corrected', reindex=False)

# hifa_flagtargets()
#
flagdata(vis='{EB1}_targets.ms', mode='list',
         inpfile='{EB1}.flagtargetscmds.txt', action='apply', flagbackup=False)

# hif_selfcal()
#
# Run self-calibration using the science target visibilities.
#
flagmanager(vis='{EB1}_targets.ms', mode='save',
            versionname='before_hif_selfcal')
flagdata(vis='{EB1}_targets.ms', mode='manual', spw='25:0~18,27:0~64',
         field='AS_209')
mstransform(vis='{EB1}_targets.ms', outputvis='{EB1}_targets.ms',
            field='AS_209', datacolumn='data', chanaverage=True, chanbin=[120, 16])
mstransform(vis='{EB2}_targets.ms', outputvis='{EB2}_targets.ms',
            field='AS_209', datacolumn='data', chanaverage=True, chanbin=[120, 16])
flagmanager(vis='{EB1}_targets.ms', mode='restore',
            versionname='before_hif_selfcal')
flagmanager(vis='{EB1}_targets.ms', mode='save',
            versionname='selfcal_starting_flags')
tclean(vis=['{EB1}_targets.ms',
       '{EB2}_targets.ms'], field='AS_209',
       imagename='Target_AS_209_Band_7_inf_EB_0', phasecenter='ICRS 16:49:15.2906
       -014.22.09.223', usemask='auto-
       multithresh', niter=0, savemodel='modelcolumn', parallel=True)
gaincal(vis='{EB1}_targets.ms',
        caltable='Target_AS_209_{EB1}_targets.ms_Band_7_inf_EB_0_p.g',
        field='AS_209', spw='25,27', solint='inf', combine='scan', calmode='p')
rmtree(path='test_inf_EB.g')
move(src='test_inf_EB.g',
     dst='Target_AS_209_{EB2}_targets.ms_Band_7_inf_EB_0_p.g')
copytree(src='a.image', dst='b.image')
applycal(vis='{EB1}_targets.ms', field='AS_209', spw='25,27',
         gaintable=['Target_AS_209_{EB1}_targets.ms_Band_7_inf_EB_0_p.g'],
         interp=['linearPD'], spwmap=[[]], calwt=[False], applymode='calflag')
applycal(vis='{EB1}_targets.ms', field='AS_209', spw='25,27',
         gaintable=['Target_AS_209_{EB1}_targets.ms_Band_7_inf_EB_0_p.g',
         'Target_AS_209_{EB1}_targets.ms_Band_7_inf_1_p.g'],
         interp=['linearPD', 'linearPD'], spwmap=[[], [0, 1, 25, 25]],
         calwt=[False], applymode='calflag')
applycal(vis='{EB2}_targets.ms', field='AS_209', spw='25,27',
         gaintable=['Target_AS_209_{EB2}_targets.ms_Band_7_inf_EB_0_p.g'],
         applymode='calflag')
tclean(vis=['{EB1}_targets.ms', '{EB2}_targets.ms'],
       imagename='Target_AS_209_Band_7_final', datacolumn='corrected',
       parallel=True)
flagdata(vis='{EB1}_targets.ms', mode='summary')
applycal(vis='{EB1}_targets.ms', field='AS_209', docallib=True,
         callib='{EB1}_targets.ms.s40.1.callibrary', applymode='calflag')
flagdata(vis='{EB1}_targets.ms', mode='summary')
applycal(vis='{EB1}_targets_line.ms', field='AS_209', docallib=True,
         callib='{EB1}_targets_line.ms.s40.2.callibrary', applymode='calflag')

# hif_makeimlist(specmode='mfs', datatype='selfcal')
"""


@pytest.mark.unit
def test_the_log_parses_into_stages_and_rejoins_wrapped_strings():
    stages = replay.parse_casa_commands(LOG)
    assert [s.name for s in stages] == [
        "hif_mstransform",
        "hifa_flagtargets",
        "hif_selfcal",
        "hif_makeimlist",
    ]
    calls = replay.stage_calls(stages, "hif_selfcal")
    tclean = next(c for c in calls if c.task == "tclean")
    # A word broken at its hyphen joins without a space, one broken at a space with one.
    assert tclean.kwargs["usemask"] == "auto-multithresh"
    assert tclean.kwargs["phasecenter"] == "ICRS 16:49:15.2906 -014.22.09.223"
    assert tclean.kwargs["vis"] == [f"{EB1}_targets.ms", f"{EB2}_targets.ms"]
    assert next(c for c in calls if c.task == "rmtree").kwargs == {"path": "test_inf_EB.g"}
    assert (
        tclean.line
        == LOG.splitlines().index(
            next(line for line in LOG.splitlines() if line.startswith("tclean("))
        )
        + 1
    )


@pytest.mark.unit
def test_the_plan_runs_each_call_where_the_pipeline_did(tmp_path):
    template = tmp_path / f"{EB1}.flagtargetstemplate.txt"
    template.write_text("# none of these\nmode='manual' antenna='DA41'\n")
    calibrated = {EB1: "/cal/a.ms.split.cal", EB2: "/cal/b.ms.split.cal"}
    plan = replay.plan_replay(
        replay.parse_casa_commands(LOG), calibrated, tmp_path / "w", {EB1: str(template)}
    )
    targets, scratch = str(tmp_path / "w" / "targets"), str(tmp_path / "w" / "selfcal")
    first = plan.actions[0]
    # The targets come from the calibrated MS's DATA, the pipeline's CORRECTED_DATA.
    assert first.task == "mstransform" and first.kwargs["vis"] == calibrated[EB1]
    assert first.kwargs["outputvis"] == f"{targets}/{EB1}_targets.ms"
    assert first.kwargs["datacolumn"] == "data"
    flags = plan.actions[2]
    assert flags.task == "flagdata" and flags.kwargs["inpfile"] == ["mode='manual' antenna='DA41'"]
    assert plan.notes == [f"{EB1}: 1 target flagging command(s)"]

    selfcal = plan.actions[3:]
    vis = [(a.task, a.kwargs.get("vis"), a.kwargs.get("versionname")) for a in selfcal]
    # Before the averaging, and the restore of the flags saved then: the full data.
    assert vis[0] == ("flagmanager", f"{targets}/{EB1}_targets.ms", "before_hif_selfcal")
    assert vis[1][1] == f"{targets}/{EB1}_targets.ms"
    assert selfcal[2].kwargs["outputvis"] == f"{scratch}/{EB1}_targets.ms"
    assert vis[4] == ("flagmanager", f"{targets}/{EB1}_targets.ms", "before_hif_selfcal")
    # After it: the averaged copies in the self-calibration's own folder.
    assert vis[5] == ("flagmanager", f"{scratch}/{EB1}_targets.ms", "selfcal_starting_flags")
    tclean = next(a for a in selfcal if a.task == "tclean")
    assert tclean.kwargs["vis"] == [f"{scratch}/{EB1}_targets.ms", f"{scratch}/{EB2}_targets.ms"]
    assert tclean.kwargs["parallel"] is False and tclean.cwd == scratch
    assert next(a for a in selfcal if a.task == "move").kwargs["src"] == "test_inf_EB.g"
    # The final image, the summaries and the final application are not replayed.
    assert not any(str(a.kwargs.get("imagename", "")).endswith("_final") for a in selfcal)
    assert sum(a.task == "flagdata" for a in selfcal) == 1
    assert not any(a.kwargs.get("docallib") for a in selfcal)
    assert plan.finally_applied == [EB1]


@pytest.mark.unit
def test_the_recipe_is_the_last_applycal_of_each_block_the_pipeline_applied(tmp_path):
    plan = replay.plan_replay(replay.parse_casa_commands(LOG), {EB1: "a", EB2: "b"}, tmp_path / "w")
    recipe = replay.apply_recipe(plan)
    # EB2's tables were never applied to its data (no final applycal): no recipe.
    assert list(recipe) == [EB1]
    (entry,) = recipe[EB1]
    assert entry["gaintable"] == [
        f"Target_AS_209_{EB1}_targets.ms_Band_7_inf_EB_0_p.g",
        f"Target_AS_209_{EB1}_targets.ms_Band_7_inf_1_p.g",
    ]
    assert entry["spwmap"] == [[], [0, 1, 25, 25]] and entry["field"] == "AS_209"


@pytest.mark.unit
def test_a_windows_spwmap_points_at_the_table_window_it_took():
    # Window 27 of a table solved over all windows together (combine='spw') takes
    # window 25's solutions; without a map a window takes its own.
    maps = [[], [0, 1, 2, 3] + list(range(4, 25)) + [25, 26, 25]]
    assert replay.window_spwmap(maps, 2, 27) == [[27], [25]]
    assert replay.window_spwmap([], 3, 25) == [[25], [25], [25]]


@pytest.mark.unit
def test_a_task_drops_the_arguments_its_casa_does_not_take():
    def tclean(vis, imagename, parallel=False):
        return None

    kept, dropped = replay._known(tclean, {"vis": "a", "imagename": "b", "fullsummary": True})
    assert kept == {"vis": "a", "imagename": "b"} and dropped == ["fullsummary"]


def _replay_folder(root: Path) -> Path:
    folder = root / "images" / replay.FOLDER
    folder.mkdir(parents=True)
    (folder / replay.APPLY_FILE).write_text(
        json.dumps(
            {
                "execution_blocks": {
                    EB1: [{"field": "AS_209", "gaintable": ["t1.g"], "spwmap": []}],
                    EB2: [{"field": "Other", "gaintable": ["t2.g"], "spwmap": []}],
                }
            }
        )
    )
    return folder


@pytest.mark.unit
def test_a_member_row_finds_its_replay_and_each_blocks_entries(tmp_path):
    member = tmp_path / "proj" / "member.uid___A001_X1_X2"
    folder = _replay_folder(member)
    row = pd.Series(
        {
            "member_folder": "proj/member.uid___A001_X1_X2",
            "member_ous_uid": "uid://A001/X1/X2",
            "field_name": "AS_209",
            "member_windows": json.dumps(
                [[f"/cal/{EB1}.ms.split.cal", 25, [3]], [f"/cal/{EB2}.ms.split.cal", 25, [3]]]
            ),
        }
    )
    entries = member_selfcal(row, tmp_path / "out", tmp_path)
    assert entries is not None
    assert entries[EB1][0]["gaintable"] == [str(folder / "t1.g")]
    assert entries[EB2] == []  # its tables are of another field
    # No replay anywhere: None, and the member is combined as it is.
    assert member_selfcal(row, tmp_path / "out", tmp_path / "elsewhere") is None
    # A replay written to an output folder (a member whose images/ is not writable).
    written = tmp_path / "out" / "member.uid___A001_X1_X2" / replay.FOLDER
    written.mkdir(parents=True)
    (written / replay.APPLY_FILE).write_text(json.dumps({"execution_blocks": {}}))
    found = replay.find_apply_file(None, "uid://A001/X1/X2", tmp_path / "out")
    assert found == written / replay.APPLY_FILE


@pytest.mark.unit
def test_a_copy_of_an_image_product_tclean_did_not_write_is_skipped(tmp_path):
    (tmp_path / "test_inf_EB.g").mkdir()
    logged = []
    replay.run_actions(
        [
            replay.Action(
                "copytree", {"src": "a_post.gridwt", "dst": "b.gridwt"}, 1, str(tmp_path)
            ),
            replay.Action("move", {"src": "test_inf_EB.g", "dst": "t.g"}, 2, str(tmp_path)),
        ],
        tasks=None,
        log=logged.append,
    )
    assert (tmp_path / "t.g").is_dir() and not (tmp_path / "b.gridwt").exists()
    assert logged[0] == "line 1 copytree: no a_post.gridwt, skipped"
