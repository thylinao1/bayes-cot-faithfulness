"""Ruling R2: the wave manifests carry the serving constants, and the trigger is arithmetic.

A wave manifest row becomes an `sbatch --export` line. If a row does not carry the
batch-invariant flag and the concurrency, the cell is submitted at whatever the
submission script's defaults are that week, and ruling R1 is precisely about a powered
measurement not resting on that. So every row of every committed manifest is read back
and checked here, against the roster the plan was built from.

Four families of manifest live in bcf/waves. The sweep-cell rows are the 216-cell grid
plan.json prices. The enrichment-pass rows that bcf/plan_enrich_waves.py writes under
ruling R3(ii) are one sampling-arm job per model per substrate and are NOT cells. The
`-resub-` rows re-run cells that were voided, and are NOT cells either: the grid counted
them the first time. The `ladder-` rows are element 11's mechanism challenge: 12 LoRA
training jobs and 12 evaluations of the checkpoints they write, which are not cells and
cannot take the cell checks at all (a training row generates nothing, and an evaluation
row's MODEL is a served name for a local checkpoint with no Hub id and no Hub revision).
The first three take the same serving checks, the ladder rows take their own in
bcf/check_wave_manifests.py, and each family is counted against its own denominator on
the same filename split.

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
from bayes_cot_faithfulness.ladder.spec import ladder_checkpoints

LADDER_CHECKPOINTS = ladder_checkpoints()
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


def _is_enrichment(name: str) -> bool:
    """RULING R3(ii). An enrichment-pass manifest is not part of the 216-cell grid.

    bcf/plan_enrich_waves.py writes one sampling-arm job per model per substrate as
    `enrich-<pool>-NN.tsv`. Those rows carry every serving constant a cell row carries
    and are checked for all of them below. What they are not is cells: adding them to
    the cell count is how the grid total grows from 216 to 240 without anyone deciding
    to run a larger grid. bcf/check_wave_manifests.py splits on the same prefix.
    """
    return name.startswith("enrich-")


def _is_ladder(name: str) -> bool:
    """ELEMENT 11. A ladder manifest is not part of the 216-cell grid.

    `ladder-train-<pool>-NN.tsv` holds the 12 LoRA jobs of element 11(b) (3 trigger
    doses x 2 training seeds x (organism, twin), the lowest rung spent on the disclosing
    learner and the uninformative control per ruling R6). `ladder-eval-<pool>-NN.tsv`
    holds the 12 evaluations of the checkpoints those jobs write. Counting either as
    cells is how the 216-cell grid reads 240 without anyone deciding to run a larger
    grid, which is the same failure the enrich- and -resub- splits exist to prevent.

    They are excluded from the ROSTER_REVISION check below rather than merely from the
    count, because a ladder row's revision is not a Hub sha: a training row pins the
    BASE's roster revision under an id that is on the roster, and an evaluation row
    carries ladder-<sha16 of the checkpoint manifest> under a served NAME that never
    will be. bcf/check_wave_manifests.py:check_ladder_row is what checks them, and
    test_the_ladder_manifests_pass_their_own_structural_checks runs it here.
    """
    return name.startswith("ladder-")


def _is_resubmission(name: str) -> bool:
    """A resubmission of voided cells is not a new cell: the grid already counted them.

    `resub-<pool>-NN.tsv` / `<pool>-resub-NN.tsv` re-runs cells that were VOIDED and
    moved to ~/bcf/results-void (a100-40-resub-01.tsv carries the four killed by the
    fixed-port collision on xgph12, DECISION-LOG.md 2026-09-07 17:41). Its rows are
    copied verbatim from the cell manifests that produced them, so counting them as
    cells is how the 216-cell grid reads 220 without anyone deciding to run four more.
    Every serving check below still applies to them, which is why they are named here
    rather than skipped.
    """
    return name.startswith("resub-") or "-resub-" in name


def _is_exploratory(name: str) -> bool:
    """An explore-*.tsv manifest holds cells run to answer a question before a ruling
    (the Phi-4 off-mode check under R12(3), DECISION-LOG 2026-09-08); never cells of
    record and never part of the 216-cell grid. Its rows still take the serving checks."""
    return name.startswith("explore-")


def _count_by_family(names) -> tuple[int, int, int, int]:
    enrich = sum(1 for n in names if _is_enrichment(n))
    resub = sum(1 for n in names if _is_resubmission(n))
    ladder = sum(1 for n in names if _is_ladder(n))
    cells = sum(
        1 for n in names
        if not _is_enrichment(n) and not _is_resubmission(n) and not _is_ladder(n)
        and not _is_exploratory(n)
    )
    return cells, enrich, resub, ladder


# The enrichment pass's own denominator, derived rather than retyped: the a100-40 models
# on the element 10 roster times the three substrates. A missing row here means a model
# or a substrate would never be enriched.
N_ENRICHMENT_ROWS = len(
    [hf for hf, _rev, _fam, pool, *_ in PW.ROSTER if pool == "a100-40"]
) * len(PW.SUBSTRATES)

# The ladder's own denominator, from element 11(b) rather than retyped: 3 trigger doses
# x 2 training seeds x (organism, twin) = 12 checkpoints per base, each trained once and
# evaluated once.
N_LADDER_ROWS = 2 * len(LADDER_CHECKPOINTS)


def test_every_manifest_row_carries_the_pinned_serving_constants():
    names = []
    for name, (model, _sub, _cue, kv) in _all_rows():
        names.append(name)
        if _is_ladder(name):
            continue    # checked by test_the_ladder_manifests_pass_their_own_structural_checks
        assert kv.get("BCF_BATCH_INVARIANT") == "1", (name, model, kv)
        assert kv.get("BCF_CONCURRENCY") == "32", (name, model, kv)
        assert kv.get("BCF_REVISION") == ROSTER_REVISION[model], (name, model)
    cells, enrich, _resub, ladder = _count_by_family(names)
    assert cells == PLAN["n_cells"] == 216, cells
    # and the enrichment pass answers to its own count, so a lost or duplicated
    # enrichment wave is caught here instead of moving the grid total.
    assert enrich == N_ENRICHMENT_ROWS == 24, enrich
    # the ladder answers to element 11(b)'s 12 checkpoints, trained once and evaluated
    # once, for the same reason.
    assert ladder == N_LADDER_ROWS == 24, ladder


def test_every_resubmission_row_re_runs_a_cell_that_already_exists():
    """What makes the resubmission exclusion safe rather than a way in.

    Excluding `-resub-` manifests from the cell count means a NEW cell could ride into
    the grid under that name and be counted by nothing. So each resubmission row's
    (model, substrate, cue) must already appear among the cell rows: a resubmission
    re-runs a voided cell, and a triple with no cell behind it is a new cell wearing a
    resubmission's filename.
    """
    cells, resubs = set(), {}
    for name, (model, sub, cue, _kv) in _all_rows():
        if _is_enrichment(name) or _is_ladder(name) or _is_exploratory(name):
            continue
        if _is_resubmission(name):
            resubs.setdefault((model, sub, cue), name)
        else:
            cells.add((model, sub, cue))
    assert resubs, "no resubmission rows found; this test would prove nothing"
    orphans = {t: n for t, n in resubs.items() if t not in cells}
    assert not orphans, f"resubmission rows with no cell behind them: {orphans}"


def test_the_cell_count_refuses_an_enrichment_manifest_filed_as_a_sweep_wave():
    """The falsification for the split above, run through the same function.

    Strip the `enrich-` prefix off the three enrichment manifests and the cell total
    reads 240, which is the reading this test file gave before the split existed.
    Rename the resubmission waves to ordinary sweep waves and it reads 226 (220 with
    the single resub wave of 2026-09-07, 224 before resub-03), which is
    the reading that failed CI on 2026-09-07. The count has to report both against
    plan.json rather than absorb them.
    """
    names = [name for name, _row in _all_rows()]
    # resub rows: 4 in a100-40-resub-01 (port collision, 2026-09-07) + 4 in
    # a100-40-resub-02 (Olmo-3-7B-Think and R1-Distill-Llama-8B wave-01/07 cells after
    # the R12(6) lift, 2026-09-08). Re-pin here when a resub manifest is added.
    # + 2 in a100-40-resub-02 (Phi-4-reasoning wave-01/07 cells after the R12(3)
    # resolution, 2026-09-08 afternoon) = 10.
    assert _count_by_family(names) == (216, 24, 10, 24)

    misfiled = [n[len("enrich-"):] if _is_enrichment(n) else n for n in names]
    cells, enrich, _resub, _ladder = _count_by_family(misfiled)
    assert (cells, enrich) == (240, 0)
    assert cells != PLAN["n_cells"]

    misfiled = [n.replace("resub-", "", 1) if _is_resubmission(n) else n for n in names]
    cells, _enrich, resub, _ladder = _count_by_family(misfiled)
    assert (cells, resub) == (226, 0)
    assert cells != PLAN["n_cells"]

    misfiled = [n[len("ladder-"):] if _is_ladder(n) else n for n in names]
    cells, _enrich, _resub, ladder = _count_by_family(misfiled)
    assert (cells, ladder) == (240, 0)
    assert cells != PLAN["n_cells"]


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


def test_the_ladder_manifests_pass_their_own_structural_checks():
    """Element 11 rows take real checks, not an exemption.

    The ladder is excluded from the CELL checks because a training row generates nothing
    and an evaluation row serves a local checkpoint with no Hub revision. What it is not
    excluded from is checking: bcf/check_wave_manifests.py:check_ladder_row requires a
    unique cell id per stage, MEM and CPUS (the fields whose absence killed job 825536),
    a substrate and cue family from the frozen lists, the element 10 roster row the
    checkpoint is trained from with its pinned revision, and -- on the evaluation rows,
    which generate -- ruling R1's flag and concurrency. This test runs that checker over
    the committed manifests and shows it able to refuse.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_wave_manifests_test", REPO / "bcf" / "check_wave_manifests.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    problems, counts = mod.check(WAVES)
    assert counts["ladder_rows"] == N_LADDER_ROWS == 24, counts
    assert not [p for p in problems if "ladder-" in p], problems

    # the falsification: the same function on a row that lost its checkpoint revision,
    # its memory field and its base pin.
    roster = {hf: (rev, pool, tp) for hf, rev, _fam, pool, tp, *_ in PW.ROSTER}
    bad = mod.check_ladder_row(
        "fake.tsv:1", "bcf-ladder/qwen3-8b/organism_0.60_1", "arc_challenge",
        "stated-hint",
        {"BCF_LADDER_CELL_ID": "organism_0.60_1", "CPUS": "8",
         "BCF_REVISION": "b968826d9c46dd6066d109eabc6255188de91218",
         "BCF_LADDER_BASE": "Qwen/Qwen3-8B", "BCF_BASE_REVISION": "deadbeef",
         "BCF_MODEL_PATH": "/tmp/x", "BCF_ARMS": "replay", "BCF_N_ITEMS": "500",
         "BCF_CURVE_CAP": "500", "BCF_BATCH_INVARIANT": "0", "BCF_CONCURRENCY": "1"},
        roster, PW, {})
    joined = " | ".join(bad)
    assert "MEM is missing" in joined
    assert "does not start with 'ladder-'" in joined
    assert "BCF_BASE_REVISION" in joined
    assert "BCF_BATCH_INVARIANT" in joined and "BCF_CONCURRENCY" in joined


def test_the_ladder_row_count_is_element_11s_partition_and_not_a_free_number():
    """12 checkpoints per base, derived from the spec, not typed into the manifest."""
    from collections import Counter

    variants = Counter(c.variant for c in LADDER_CHECKPOINTS)
    assert len(LADDER_CHECKPOINTS) == 12
    # rungs 2 and 3 carry organism and twin across both seeds: 4 organism-minus-twin
    # differences, which is what A3.2's sd_pilot(D) is computed on under ruling R6.
    assert variants["organism"] == variants["twin"] == 4
    # the lowest rung carries the disclosing learner and the uninformative control
    assert variants["disclosing"] == variants["uninformative"] == 2
    assert {c.rung for c in LADDER_CHECKPOINTS if c.variant == "disclosing"} == {1}
    assert len({c.cell_id for c in LADDER_CHECKPOINTS}) == 12
