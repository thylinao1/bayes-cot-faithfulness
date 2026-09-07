"""Ruling R3(ii)/(iii): the item-list path, the enrichment flag, and the list file.

Three layers, each tested where it lives:

  * ``bayes_cot_faithfulness.item_list`` resolves a list file against a pool. Pure
    arithmetic on lists; every refusal is asserted by the message it names.
  * ``experiments/08_additive_arms.py --item-list`` runs on exactly the named items,
    marks every record ``enrichment=true``, and reports the two populations apart. Run
    end to end against the same scripted offline client the resume tests use, so a full
    ``run()`` executes with no network and no model server.
  * ``bcf/enrich_report.py`` turns a finished sampling pass into the list file the cell
    reads. The round trip is asserted: what the report writes, the resolver accepts.

The threshold cases are hand-computed. At k = 4 with a 4-option item, an answer split of
3/1 gives Shannon entropy -(0.75 ln 0.75 + 0.25 ln 0.25) = 0.562335 nats and a normalized
entropy of 0.562335 / ln 4 = 0.405639, which is ABOVE the frozen 0.30; a 4/0 split gives
exactly 0.0, which is below it. Those two straddle the flag and are what the fixtures use.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from bayes_cot_faithfulness.interventions import QAItem
from bayes_cot_faithfulness.item_list import (
    ItemListError,
    parse_item_list,
    question_sha16,
    select_items,
)
from bayes_cot_faithfulness.sampling_arm import (
    UNCERTAIN_ENTROPY_THRESHOLD,
    summarize_item_samples,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "08_additive_arms.py"
REPORT_SCRIPT = REPO / "bcf" / "enrich_report.py"
HOST = "http://localhost:11434"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load(SCRIPT, "arms_under_test_enrichment")
report_mod = _load(REPORT_SCRIPT, "enrich_report_under_test")


# --------------------------------------------------------------------------- #
# Fixtures: a small pool and an offline client
# --------------------------------------------------------------------------- #
def _mcq(question: str) -> dict:
    return {"question": question, "choices": ["alpha", "bravo", "charlie", "delta"],
            "answer_index": 0}


def _pool(tmp_path: Path, n: int = 6):
    rows = [_mcq(f"pick the label for gadget number {i:02d}") for i in range(n)]
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(rows))
    items = [QAItem(r["question"], tuple(r["choices"]), r["answer_index"]) for r in rows]
    return path, rows, items


def _holdout(tmp_path: Path):
    rows = [_mcq(f"pick the label for widget number {i:02d}") for i in range(2)]
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(rows))
    return path


class ScriptedClient:
    """Every generation parses; every clean answer is correct. Counts its calls."""

    def __init__(self, items):
        self._label = {it.question: it.answer_label for it in items}
        self.calls = 0
        self.seen_questions: list[str] = []

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, num_predict: int = 0) -> str:
        self.calls += 1
        matches = [q for q in self._label if q in prompt]
        if not matches:
            raise AssertionError(f"prompt matched no known item: {prompt[:80]!r}")
        question = max(matches, key=len)
        self.seen_questions.append(question)
        return f"1. scripted reasoning step\nAnswer: ({self._label[question]})"


def _list_file(path: Path, indices, rows, *, pool_path=None, extra=None):
    payload = {
        "schema": "bcf.item_list.v1",
        "selector": "right_but_uncertain",
        "pool_size": len(rows),
        "items": [{"index": i, "question_sha16": question_sha16(rows[i]["question"])}
                  for i in indices],
    }
    if pool_path is not None:
        import hashlib
        payload["pool_sha256"] = hashlib.sha256(Path(pool_path).read_bytes()).hexdigest()
    payload.update(extra or {})
    path.write_text(json.dumps(payload))
    return path


# --------------------------------------------------------------------------- #
# (1) The resolver
# --------------------------------------------------------------------------- #
def test_parse_accepts_the_written_file_and_a_bare_array(tmp_path):
    _, rows, _ = _pool(tmp_path)
    full = parse_item_list({"items": [{"index": 2, "question_sha16": "ab"}],
                            "selector": "right_but_uncertain"})
    assert full["entries"] == [{"index": 2, "question_sha16": "ab"}]
    assert full["meta"]["selector"] == "right_but_uncertain"
    bare = parse_item_list([0, 3])
    assert bare["entries"] == [{"index": 0, "question_sha16": None},
                               {"index": 3, "question_sha16": None}]
    assert bare["meta"] == {}
    assert len(rows) == 6


def test_parse_refuses_an_empty_list_a_duplicate_and_a_non_integer():
    with pytest.raises(ItemListError, match="selects no items"):
        parse_item_list([])
    with pytest.raises(ItemListError, match="appears twice"):
        parse_item_list([1, 1])
    with pytest.raises(ItemListError, match="neither an integer nor an object"):
        parse_item_list(["3"])
    with pytest.raises(ItemListError, match="no 'items' array"):
        parse_item_list({"selector": "right_but_uncertain"})


def test_select_returns_the_named_items_in_list_order(tmp_path):
    _, rows, items = _pool(tmp_path)
    parsed = parse_item_list({"items": [
        {"index": 4, "question_sha16": question_sha16(rows[4]["question"])},
        {"index": 1, "question_sha16": question_sha16(rows[1]["question"])},
    ]})
    got, report = select_items(items, parsed["entries"], meta=parsed["meta"])
    assert [it.question for it in got] == [rows[4]["question"], rows[1]["question"]]
    assert report["n_selected"] == 2
    assert report["n_verified_by_question_hash"] == 2
    assert report["n_unverified"] == 0
    assert report["indices"] == [4, 1]


def test_select_refuses_an_index_outside_the_pool(tmp_path):
    _, _rows, items = _pool(tmp_path)
    parsed = parse_item_list([99])
    with pytest.raises(ItemListError, match="outside the 6-item pool"):
        select_items(items, parsed["entries"], meta=parsed["meta"])


def test_select_refuses_when_the_question_hash_does_not_match(tmp_path):
    _, rows, items = _pool(tmp_path)
    # The index is valid and the hash belongs to a DIFFERENT item: exactly what a
    # refetched pool looks like when the order moved.
    parsed = parse_item_list([{"index": 0,
                               "question_sha16": question_sha16(rows[3]["question"])}])
    with pytest.raises(ItemListError, match="The pool moved under the list"):
        select_items(items, parsed["entries"], meta=parsed["meta"])


def test_select_refuses_a_pool_hash_or_size_that_disagrees(tmp_path):
    _, _rows, items = _pool(tmp_path)
    parsed = parse_item_list({"items": [0], "pool_sha256": "deadbeef"})
    with pytest.raises(ItemListError, match="different pool file"):
        select_items(items, parsed["entries"], pool_sha256="cafe", meta=parsed["meta"])
    parsed = parse_item_list({"items": [0], "pool_size": 1500})
    with pytest.raises(ItemListError, match="the indices do not carry across"):
        select_items(items, parsed["entries"], meta=parsed["meta"])


def test_a_bare_index_list_is_resolved_but_reported_as_unverified(tmp_path):
    _, rows, items = _pool(tmp_path)
    parsed = parse_item_list([2, 5])
    got, report = select_items(items, parsed["entries"], meta=parsed["meta"])
    assert [it.question for it in got] == [rows[2]["question"], rows[5]["question"]]
    assert report["n_verified_by_question_hash"] == 0
    assert report["n_unverified"] == 2


# --------------------------------------------------------------------------- #
# (2) The runner's item-list path and the enrichment flag
# --------------------------------------------------------------------------- #
def _run(monkeypatch, client, *, data_path, out_dir, holdout_path, item_list=None,
         arms=("replay",), n_items=6):
    monkeypatch.setattr(mod, "_gate_client", lambda *a, **k: client)
    return mod.run("fake", HOST, n_items, data_path, out_dir, list(arms),
                   curve_cap=n_items, num_predict=320, backend="ollama",
                   specificity_holdout=holdout_path, item_list=item_list)


def test_the_cell_runs_on_exactly_the_listed_items_and_marks_them_enriched(
        tmp_path, monkeypatch):
    pool_path, rows, items = _pool(tmp_path)
    holdout = _holdout(tmp_path)
    # Three items, not two: the runner refuses to run any arm on fewer than three
    # clean-correct records, so a real enrichment cell whose stratum is thinner than that
    # never starts. That floor is asserted in its own test below.
    lst = _list_file(tmp_path / "uncertain_items.json", [4, 1, 5], rows,
                     pool_path=pool_path)
    client = ScriptedClient(items)
    out = tmp_path / "cell"
    assert _run(monkeypatch, client, data_path=pool_path, out_dir=out,
                holdout_path=holdout, item_list=lst) == 0

    transcripts = json.loads((out / "arms_transcripts_fake.json").read_text())
    assert [t["question"] for t in transcripts] == [rows[4]["question"],
                                                    rows[1]["question"],
                                                    rows[5]["question"]]
    assert all(t["enrichment"] is True for t in transcripts)
    # No item outside the list was ever put to the model.
    assert set(client.seen_questions) == {rows[4]["question"], rows[1]["question"],
                                          rows[5]["question"]}

    summary = json.loads((out / "arms_summary_fake.json").read_text())
    assert summary["n_items"] == 3
    e = summary["enrichment"]
    assert e["n_records_enrichment"] == 3
    assert e["n_records_regular"] == 0
    assert e["n_clean_correct_enrichment"] == 3
    assert e["n_clean_correct_regular"] == 0
    assert e["item_list"]["n_selected"] == 3
    assert e["item_list"]["n_verified_by_question_hash"] == 3
    assert e["item_list"]["indices"] == [4, 1, 5]
    assert e["item_list"]["selector"] == "right_but_uncertain"


def test_a_list_thinner_than_three_items_runs_no_arm(tmp_path, monkeypatch, capsys):
    """The runner's own floor, stated here because it decides what an enrichment cell is.

    A model whose uncertain stratum on a substrate holds one or two items cannot have an
    enrichment cell at all: the runner stops after the clean pass. That is ruling R3(iii)
    arriving early, at the cell rather than at the verdict, and it is better seen here
    than as a mysterious empty results directory on the cluster.
    """
    pool_path, rows, items = _pool(tmp_path)
    holdout = _holdout(tmp_path)
    lst = _list_file(tmp_path / "thin.json", [0, 2], rows, pool_path=pool_path)
    out = tmp_path / "cell"
    assert _run(monkeypatch, ScriptedClient(items), data_path=pool_path, out_dir=out,
                holdout_path=holdout, item_list=lst) == 0
    assert not (out / "arms_summary_fake.json").exists()
    assert "too few clean-correct items" in capsys.readouterr().out


def test_a_regular_cell_is_untouched_and_reports_a_zero_enrichment_count(
        tmp_path, monkeypatch):
    pool_path, _rows, items = _pool(tmp_path)
    holdout = _holdout(tmp_path)
    out = tmp_path / "cell"
    assert _run(monkeypatch, ScriptedClient(items), data_path=pool_path, out_dir=out,
                holdout_path=holdout) == 0
    transcripts = json.loads((out / "arms_transcripts_fake.json").read_text())
    assert len(transcripts) == 6
    assert all(t["enrichment"] is False for t in transcripts)
    e = json.loads((out / "arms_summary_fake.json").read_text())["enrichment"]
    assert e["n_records_enrichment"] == 0
    assert e["n_records_regular"] == 6
    assert e["item_list"] is None


def test_a_stale_list_refuses_before_any_model_call(tmp_path, monkeypatch, capsys):
    pool_path, rows, items = _pool(tmp_path)
    holdout = _holdout(tmp_path)
    bad = tmp_path / "stale.json"
    bad.write_text(json.dumps({"items": [
        {"index": 0, "question_sha16": question_sha16(rows[3]["question"])}]}))
    client = ScriptedClient(items)
    out = tmp_path / "cell"
    assert _run(monkeypatch, client, data_path=pool_path, out_dir=out,
                holdout_path=holdout, item_list=bad) == 0
    assert client.calls == 0
    assert not (out / "arms_summary_fake.json").exists()
    assert not (out / "arms_checkpoint_fake.json").exists()
    printed = capsys.readouterr().out
    assert "REFUSED" in printed
    assert "The pool moved under the list" in printed


def test_a_missing_list_file_refuses_before_any_model_call(tmp_path, monkeypatch, capsys):
    pool_path, _rows, items = _pool(tmp_path)
    holdout = _holdout(tmp_path)
    client = ScriptedClient(items)
    assert _run(monkeypatch, client, data_path=pool_path, out_dir=tmp_path / "cell",
                holdout_path=holdout, item_list=tmp_path / "nope.json") == 0
    assert client.calls == 0
    assert "unreadable list file" in capsys.readouterr().out


def test_the_item_list_is_part_of_the_resume_fingerprint(tmp_path, monkeypatch):
    """A second leg pointed at a DIFFERENT list must refuse, not merge two populations."""
    pool_path, rows, items = _pool(tmp_path)
    holdout = _holdout(tmp_path)
    first = _list_file(tmp_path / "a.json", [0, 1], rows, pool_path=pool_path)
    second = _list_file(tmp_path / "b.json", [2, 3], rows, pool_path=pool_path)
    out = tmp_path / "cell"
    assert _run(monkeypatch, ScriptedClient(items), data_path=pool_path, out_dir=out,
                holdout_path=holdout, item_list=first) == 0
    saved = json.loads((out / "arms_checkpoint_fake.json").read_text())
    fingerprint = saved["params"]["item_list"]
    assert fingerprint["n_selected"] == 2
    assert fingerprint["indices"] == [0, 1]

    import experiments.arms_resume as _unused  # noqa: F401  (import path sanity)
    mismatches = mod.arms_resume.params_mismatch(
        saved["params"], dict(saved["params"], item_list={"path": str(second)}))
    assert [m[0] for m in mismatches] == ["item_list"]


# --------------------------------------------------------------------------- #
# (3) The enrichment report: from a finished sampling pass to the list file
# --------------------------------------------------------------------------- #
def _sampling_block(answers, *, answer_label="A"):
    return summarize_item_samples(answers, n_options=4, answer_label=answer_label,
                                  item_labels=["A", "B", "C", "D"],
                                  threshold=UNCERTAIN_ENTROPY_THRESHOLD, k_grid=(4,))


def test_the_report_writes_a_list_of_exactly_the_uncertain_items(tmp_path):
    pool_path, rows, _items = _pool(tmp_path)
    run_dir = tmp_path / "pass"
    run_dir.mkdir()
    # NOTE index 0: 3/1 split, mode correct  -> normalized entropy 0.405639, UNCERTAIN
    # index 1: 4/0 split, mode correct  -> entropy 0.0, confident
    # index 2: 3/1 split, mode WRONG    -> uncertain but not right; excluded
    # index 3: no sampling block at all -> never scored; excluded and counted
    def _row(i, sampling=None):
        out = {"question": rows[i]["question"], "choices": rows[i]["choices"],
               "answer_label": "A", "clean_correct": sampling is not None}
        if sampling is not None:
            out["sampling"] = sampling
        return out

    transcripts = [
        _row(0, _sampling_block(["A", "A", "A", "B"])),
        _row(1, _sampling_block(["A", "A", "A", "A"])),
        _row(2, _sampling_block(["B", "B", "B", "A"])),
        _row(3),
    ]
    (run_dir / "arms_transcripts_fake.json").write_text(json.dumps(transcripts))

    # The two entropies the fixture straddles the threshold with, computed from the
    # frozen definition rather than read back out of the implementation.
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(4)
    assert transcripts[0]["sampling"]["normalized_entropy"] == pytest.approx(expected)
    assert expected > UNCERTAIN_ENTROPY_THRESHOLD
    assert transcripts[1]["sampling"]["normalized_entropy"] == pytest.approx(0.0)

    rc = report_mod.main(["--out-dir", str(run_dir), "--data", str(pool_path),
                          "--substrate", "toy"])
    assert rc == 0

    listed = json.loads((run_dir / "uncertain_items.json").read_text())
    assert listed["n_items"] == 1
    assert listed["items"] == [{"index": 0,
                                "question_sha16": question_sha16(rows[0]["question"])}]
    assert listed["entropy_threshold_used"] == UNCERTAIN_ENTROPY_THRESHOLD
    assert listed["pool_size"] == 6
    assert listed["substrate"] == "toy"

    report = json.loads((run_dir / "enrichment_report.json").read_text())
    assert report["n_items_scored"] == 3
    assert report["n_modal_correct"] == 2
    assert report["n_right_but_uncertain"] == 1
    assert report["accounting"]["n_without_sampling_block"] == 1

    rowsout = [json.loads(ln)
               for ln in (run_dir / "enrichment_items.jsonl").read_text().splitlines()]
    assert [r["index"] for r in rowsout] == [0, 1, 2]
    assert [r["right_but_uncertain"] for r in rowsout] == [True, False, False]
    assert [r["modal_answer"] for r in rowsout] == ["A", "A", "B"]
    assert rowsout[0]["normalized_entropy"] == pytest.approx(expected)


def test_the_written_list_resolves_against_the_pool_it_names(tmp_path):
    """The round trip: what the report writes is what the runner accepts."""
    pool_path, rows, items = _pool(tmp_path)
    run_dir = tmp_path / "pass"
    run_dir.mkdir()
    (run_dir / "arms_transcripts_fake.json").write_text(json.dumps([
        {"question": rows[2]["question"], "choices": rows[2]["choices"],
         "answer_label": "A", "clean_correct": True,
         "sampling": _sampling_block(["A", "A", "A", "B"])},
    ]))
    assert report_mod.main(["--out-dir", str(run_dir), "--data", str(pool_path)]) == 0
    payload = json.loads((run_dir / "uncertain_items.json").read_text())
    parsed = parse_item_list(payload)
    import hashlib
    got, report = select_items(
        items, parsed["entries"],
        pool_sha256=hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        meta=parsed["meta"])
    assert [it.question for it in got] == [rows[2]["question"]]
    assert report["n_verified_by_question_hash"] == 1


def test_a_pass_with_no_sampling_block_writes_nothing_and_says_so(tmp_path, capsys):
    pool_path, rows, _items = _pool(tmp_path)
    run_dir = tmp_path / "pass"
    run_dir.mkdir()
    (run_dir / "arms_transcripts_fake.json").write_text(json.dumps([
        {"question": rows[0]["question"], "choices": rows[0]["choices"],
         "answer_label": "A", "clean_correct": True},
    ]))
    assert report_mod.main(["--out-dir", str(run_dir), "--data", str(pool_path)]) == 3
    assert not (run_dir / "uncertain_items.json").exists()
    assert "NONE carries a sampling block" in capsys.readouterr().out
