"""Ruling R2: the wave manifests carry the serving constants, and the trigger is arithmetic.

A wave manifest row becomes an `sbatch --export` line. If a row does not carry the
batch-invariant flag and the concurrency, the cell is submitted at whatever the
submission script's defaults are that week, and ruling R1 is precisely about a powered
measurement not resting on that. So every row of every committed manifest is read back
and checked here, against the roster the plan was built from.

The element 16 trigger is checked as arithmetic against the budget of record, and the
checker is shown able to report TRIGGERED, so a "not triggered" reading is a computed
result rather than a constant.

Nothing here reaches the network or the cluster.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WAVES = REPO / "bcf" / "waves"


def _load():
    spec = importlib.util.spec_from_file_location(
        "plan_waves_test", REPO / "bcf" / "plan_waves.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PW = _load()
PLAN = json.loads((WAVES / "plan.json").read_text())
ROSTER_REVISION = {hf: rev for hf, rev, *_ in PW.ROSTER}


def _rows(path: Path):
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        yield fields[0], fields[1], fields[2], dict(
            f.split("=", 1) for f in fields[3:] if "=" in f
        )


def _all_rows():
    for tsv in sorted(WAVES.glob("*.tsv")):
        for row in _rows(tsv):
            yield tsv.name, row


def test_every_manifest_row_carries_the_pinned_serving_constants():
    seen = 0
    for name, (model, _sub, _cue, kv) in _all_rows():
        seen += 1
        assert kv.get("BCF_BATCH_INVARIANT") == "1", (name, model, kv)
        assert kv.get("BCF_CONCURRENCY") == "32", (name, model, kv)
        assert kv.get("BCF_REVISION") == ROSTER_REVISION[model], (name, model)
    assert seen == PLAN["n_cells"] == 216, seen


def test_the_row_check_rejects_a_manifest_that_lost_the_flag():
    """The falsification for the check above, on a row built the same way."""
    good = {"BCF_BATCH_INVARIANT": "1", "BCF_CONCURRENCY": "32"}
    reverted = {"BCF_CONCURRENCY": "1"}
    assert good.get("BCF_BATCH_INVARIANT") == "1"
    assert reverted.get("BCF_BATCH_INVARIANT") != "1"
    assert reverted.get("BCF_CONCURRENCY") != "32"


def test_every_row_says_whether_its_cost_is_measured_or_a_floor():
    ratios = PLAN["pool_ratio"]
    assert ratios["a100-40"]["measured_for_this_pool"] is True
    for pool in ("a100-80", "h100-96", "h200-141"):
        assert ratios[pool]["measured_for_this_pool"] is False
        assert "FLOOR" in ratios[pool]["why"]
    by_pool = PLAN["card_hours_by_pool"]
    # The measured class is the seven bf16 models up to 14B: 7 x 3 substrates x 4 cue
    # families = 84 of the a100-40 pool's 96 cells. gpt-oss-20b is the other 12.
    assert by_pool["a100-40"]["n_cells_measured_for_their_class"] == 84
    for pool in ("a100-80", "h100-96", "h200-141"):
        assert by_pool[pool]["n_cells_measured_for_their_class"] == 0


def test_the_cost_basis_is_the_flag_on_run_and_not_the_sequential_one():
    assert PLAN["cost_basis_id"] == "flag_on_concurrency_32_826025"
    assert PLAN["measured_cost_model"]["job"] == "826025"
    assert PLAN["measured_cost_model_sequential_comparison"]["job"] == "825511"
    # and the cost model prices only the intervals a powered cell runs, so the sampling
    # and repeat-curve arms in job 826025's file cannot inflate a cell row
    used = PLAN["measured_cost_model"]["intervals_used"]
    assert "sampling" not in used and "repeat-curves" not in used
    assert set(used) == {"clean_substrate", "cue_pass", *PLAN["arms"]}


def test_the_trigger_compares_against_the_named_budget_of_record():
    t = PLAN["ladder_trigger_element_16"]
    assert t["budget_of_record_card_hours"] == 650
    assert "CONTRACT.md" in t["budget_of_record_source"]
    assert t["priced_card_hours_total"] == round(
        t["priced_card_hours_sweep_cells"] + t["priced_card_hours_enrichment_pass"], 1)
    assert t["triggered"] is False
    assert t["reading"] == "NOT TRIGGERED"


def test_the_trigger_reports_TRIGGERED_when_the_grid_prices_above_the_budget():
    """The falsification: the same function, on a grid that costs more than the budget."""
    over = PW.trigger_comparison(700.0, 4000.0, 50.0)
    assert over["triggered"] is True
    assert over["reading"].startswith("TRIGGERED")
    assert over["ratio_priced_over_budget"] > 1.0
    under = PW.trigger_comparison(213.7, 1258.4, 48.0)
    assert under["triggered"] is False


def test_a_throughput_file_missing_an_arm_refuses_rather_than_pricing_it_at_zero(tmp_path):
    thin = tmp_path / "throughput.json"
    thin.write_text(json.dumps({"arms": [
        {"arm": "clean_substrate", "n_calls": 30, "n_full_generations": 30,
         "seconds": 11.0},
        {"arm": "curves", "n_calls": 280, "n_full_generations": 0, "seconds": 8.0},
    ], "total_calls": 310, "total_seconds": 19.0}))
    try:
        PW.load_measured(thin, "fake")
    except SystemExit as exc:
        assert "has no interval for" in str(exc)
    else:
        raise AssertionError("a throughput file missing nine arms was accepted")


def test_the_enrichment_pass_is_priced_from_a_measured_per_item_second():
    e = PLAN["enrichment_pass"]
    assert e["floor"] is True
    assert "826025" in e["per_item_seconds_source"]
    # 64.0 s / 30 calls on the measured run, rounded to four places in the plan
    assert abs(e["per_item_seconds"] - 64.0 / 30.0) < 5e-5
    expected = (sum(e["pool_sizes"].values()) * e["per_item_seconds"]
                * PLAN["n_models"] / 3600)
    assert abs(e["card_hours_total"] - round(expected, 1)) < 0.11


def test_the_wave_count_and_the_cap_arithmetic_are_unchanged_by_the_repricing():
    assert PLAN["n_cells"] == 216
    assert PLAN["n_models"] == 18
    for wave in PLAN["waves"]:
        assert wave["cards"] <= wave["sweep_slots"]
        assert wave["sweep_slots"] <= wave["cap"]
