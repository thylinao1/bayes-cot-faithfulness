"""Unit tests for experiments/fetch_logiqa2.py and experiments/fetch_aqua.py.

The two fetchers feed the P3 substrate pilots (docs/p3_substrate_scouting.md): they
must write the exact {question, choices, answer_index} schema the runners load, keep
the shipped file's line order (the pilots take "the first N items in fetch order"),
and drop -- never repair -- any row whose answer-key mapping cannot be trusted (C3:
a noisy key poisons both the bait planting and the clean-correct filter).

Everything here is offline and synthetic: the pure converters are exercised on
hand-built rows and JSONL blobs, never on a network fetch. The modules are loaded by
file path exactly like tests/test_score_labels.py loads score_labels.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from bayes_cot_faithfulness.interventions import QAItem

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


logiqa = _load("fetch_logiqa2")
aqua = _load("fetch_aqua")


def logiqa_row(**overrides) -> dict:
    row = {
        "id": 1,
        "answer": 2,
        "text": "All swans in the lake are white.",
        "question": "Which conclusion follows?",
        "options": ["opt A", "opt B", "opt C", "opt D"],
        "type": {"categorical reasoning": True},
    }
    row.update(overrides)
    return row


def aqua_row(**overrides) -> dict:
    row = {
        "question": "What is 2 + 3 * 4?",
        "options": ["A)10", "B)14", "C)20", "D)24", "E)None of these"],
        "rationale": "ignored on purpose",
        "correct": "B",
    }
    row.update(overrides)
    return row


# --- LogiQA 2.0 converter ----------------------------------------------------
def test_logiqa_valid_row_maps_schema():
    item = logiqa.to_item(logiqa_row())
    assert item == {
        "question": "All swans in the lake are white.\n\nWhich conclusion follows?",
        "choices": ["opt A", "opt B", "opt C", "opt D"],
        "answer_index": 2,
    }


def test_logiqa_passage_prepended_with_blank_line_and_stripped():
    item = logiqa.to_item(logiqa_row(text="  passage  ", question="  q?  "))
    assert item is not None
    assert item["question"] == "passage\n\nq?"


@pytest.mark.parametrize("answer", [-1, 4, "2", None, 2.0])
def test_logiqa_bad_answer_rejected(answer):
    assert logiqa.to_item(logiqa_row(answer=answer)) is None


def test_logiqa_bool_answer_rejected_despite_being_int_subclass():
    assert logiqa.to_item(logiqa_row(answer=True)) is None


@pytest.mark.parametrize(
    "options",
    [
        ["a", "b", "c"],  # three options
        ["a", "b", "c", "d", "e"],  # five options
        ["a", "b", "c", ""],  # empty option text
        ["a", "b", "c", None],  # non-string option
        "not a list",
        None,
    ],
)
def test_logiqa_bad_options_rejected(options):
    assert logiqa.to_item(logiqa_row(options=options)) is None


@pytest.mark.parametrize("field", ["text", "question"])
@pytest.mark.parametrize("value", ["", "   ", None, 7])
def test_logiqa_missing_or_blank_passage_or_question_rejected(field, value):
    assert logiqa.to_item(logiqa_row(**{field: value})) is None


# --- AQuA-RAT converter ------------------------------------------------------
def test_aqua_valid_row_strips_prefixes_and_maps_correct_letter():
    item = aqua.to_item(aqua_row())
    assert item == {
        "question": "What is 2 + 3 * 4?",
        "choices": ["10", "14", "20", "24", "None of these"],
        "answer_index": 1,
    }


def test_aqua_prefix_whitespace_variants_stripped():
    row = aqua_row(options=["A) 10", "B )14", "C)  20", "D)24", "E) None"])
    item = aqua.to_item(row)
    assert item is not None
    assert item["choices"] == ["10", "14", "20", "24", "None"]


def test_aqua_positional_prefix_mismatch_rejects_whole_row():
    # B) in first position: the option order and the correct letter can no longer
    # be trusted to agree, so the row is dropped rather than remapped.
    row = aqua_row(options=["B)10", "A)14", "C)20", "D)24", "E)None"])
    assert aqua.to_item(row) is None


def test_aqua_missing_prefix_rejects_whole_row():
    row = aqua_row(options=["10", "B)14", "C)20", "D)24", "E)None"])
    assert aqua.to_item(row) is None


def test_aqua_empty_option_text_after_prefix_rejected():
    row = aqua_row(options=["A)10", "B)14", "C)", "D)24", "E)None"])
    assert aqua.to_item(row) is None


@pytest.mark.parametrize("correct", ["F", "", None, 1, "AB"])
def test_aqua_bad_correct_letter_rejected(correct):
    assert aqua.to_item(aqua_row(correct=correct)) is None


def test_aqua_correct_letter_whitespace_tolerated():
    item = aqua.to_item(aqua_row(correct=" E "))
    assert item is not None
    assert item["answer_index"] == 4


@pytest.mark.parametrize(
    "options",
    [
        ["A)1", "B)2", "C)3", "D)4"],  # four options
        ["A)1", "B)2", "C)3", "D)4", "E)5", "F)6"],  # six options
        "not a list",
        None,
    ],
)
def test_aqua_bad_option_count_rejected(options):
    assert aqua.to_item(aqua_row(options=options)) is None


# --- JSONL parsing: order, skipping, and determinism -------------------------
@pytest.mark.parametrize("module,rowmaker", [(logiqa, logiqa_row), (aqua, aqua_row)])
def test_parse_jsonl_preserves_line_order_and_skips_junk(module, rowmaker):
    if module is logiqa:
        rows = [rowmaker(id=i, question=f"q{i}?") for i in range(3)]
    else:
        rows = [rowmaker(question=f"q{i}?") for i in range(3)]
    blob = "\n".join(
        [
            json.dumps(rows[0]),
            "",  # blank line skipped
            "not json at all {",  # malformed line skipped
            json.dumps(["a", "list"]),  # non-object skipped
            json.dumps(rows[1]),
            json.dumps(rowmaker(options=None)),  # malformed row skipped
            json.dumps(rows[2]),
        ]
    )
    items = module.parse_jsonl(blob)
    assert [it["question"] for it in items] == [
        module.to_item(r)["question"] for r in rows
    ]


@pytest.mark.parametrize("module,rowmaker", [(logiqa, logiqa_row), (aqua, aqua_row)])
def test_parse_jsonl_is_deterministic(module, rowmaker):
    blob = "\n".join(json.dumps(rowmaker(question=f"q{i}?")) for i in range(5))
    assert module.parse_jsonl(blob) == module.parse_jsonl(blob)


# --- Review-panel regressions (drop-never-repair, JSONL semantics, --n) -------
def test_logiqa_duplicate_answer_key_drops_row_instead_of_last_wins():
    # Plain json.loads would keep the LAST "answer" (probe-confirmed last-wins), so a
    # row carrying two divergent answers must be dropped whole, never trusted.
    line = ('{"id": 1, "answer": 0, "text": "p", "question": "q?", '
            '"options": ["a", "b", "c", "d"], "answer": 3}')
    assert logiqa.parse_jsonl(line) == []


def test_aqua_duplicate_options_arrays_drop_row_instead_of_remapping():
    # Two options arrays, both with valid positional prefixes: last-wins would map
    # "correct" onto whichever list the file listed last. The row must be dropped.
    line = ('{"question": "q?", "options": ["A)x1", "B)x2", "C)x3", "D)x4", "E)x5"], '
            '"correct": "B", "options": ["A)y1", "B)y2", "C)y3", "D)y4", "E)y5"]}')
    assert aqua.parse_jsonl(line) == []


def test_aqua_duplicate_correct_key_drops_row():
    line = ('{"question": "q?", "correct": "A", '
            '"options": ["A)1", "B)2", "C)3", "D)4", "E)5"], "correct": "E"}')
    assert aqua.parse_jsonl(line) == []


def test_nested_duplicate_keys_also_drop_the_row():
    # The hook applies to nested objects too (LogiQA rows carry a "type" dict);
    # drop-never-repair extends to them rather than special-casing depth.
    line = ('{"id": 1, "answer": 2, "text": "p", "question": "q?", '
            '"options": ["a", "b", "c", "d"], "type": {"k": true, "k": false}}')
    assert logiqa.parse_jsonl(line) == []


@pytest.mark.parametrize("module,rowmaker", [(logiqa, logiqa_row), (aqua, aqua_row)])
def test_row_containing_raw_u2028_in_a_string_survives(module, rowmaker):
    # U+2028 is legal unescaped inside a JSON string. splitlines() would shatter the
    # physical line and silently DROP the row, shifting "first N in fetch order";
    # splitting on \n only keeps it whole.
    row = rowmaker(question="first part\u2028second part?")
    items = module.parse_jsonl(json.dumps(row, ensure_ascii=False))
    assert len(items) == 1
    assert "\u2028" in items[0]["question"]


@pytest.mark.parametrize("module,rowmaker", [(logiqa, logiqa_row), (aqua, aqua_row)])
def test_object_embedded_after_u2028_junk_is_not_parsed_as_a_row(module, rowmaker):
    # Under splitlines(), 'junk<U+2028>{valid row}' had its embedded object ACCEPTED
    # as a standalone row; under \n-only splitting the whole physical line is one
    # undecodable line and is dropped.
    blob = "junk\u2028" + json.dumps(rowmaker(), ensure_ascii=False)
    assert module.parse_jsonl(blob) == []


@pytest.mark.parametrize("module", [logiqa, aqua])
@pytest.mark.parametrize("bad_n", ["0", "-120"])
def test_non_positive_n_is_rejected_before_any_download(module, bad_n):
    # A negative --n would slice [: -n] and silently keep all-but-the-last items.
    # ap.error exits 2 before _download is ever reached.
    with pytest.raises(SystemExit) as exc:
        module.main(["--n", bad_n])
    assert exc.value.code == 2


@pytest.mark.parametrize("module", [logiqa, aqua])
def test_raw_url_is_commit_pinned_not_branch_tracking(module):
    assert module.PINNED_COMMIT in module.RAW_URL
    assert len(module.PINNED_COMMIT) == 40
    assert "/main/" not in module.RAW_URL and "/master/" not in module.RAW_URL
    assert module.RAW_URL.startswith("https://")


@pytest.mark.parametrize("module,rowmaker", [(logiqa, logiqa_row), (aqua, aqua_row)])
def test_main_end_to_end_offline_tolerates_bom(module, rowmaker, tmp_path, monkeypatch):
    # Drives main() with a stubbed download (no network): a BOM-prefixed payload must
    # not silently lose its first row (utf-8-sig decode), and the written file must
    # hold the runner schema in fetch order.
    rows = [rowmaker(question=f"q{i}?") for i in range(3)]
    payload = "\ufeff" + "\n".join(json.dumps(r) for r in rows)
    monkeypatch.setattr(module, "_download", lambda split: payload.encode("utf-8"))
    out = tmp_path / "items.json"
    assert module.main(["--n", "2", "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert [it["question"] for it in written] == [
        module.to_item(r)["question"] for r in rows[:2]
    ]
    assert set(written[0]) == {"question", "choices", "answer_index"}


# --- Frozen-instrument compatibility (C1) ------------------------------------
@pytest.mark.parametrize(
    "module,rowmaker,n_choices",
    [(logiqa, logiqa_row, 4), (aqua, aqua_row, 5)],
)
def test_items_construct_valid_qaitems_within_frozen_label_class(
    module, rowmaker, n_choices
):
    """The emitted schema must load into QAItem with labels inside the frozen [A-F]
    capture class -- the C1 criterion the substrate memo fixes in advance."""
    item = module.to_item(rowmaker())
    qa = QAItem(
        question=item["question"],
        choices=tuple(item["choices"]),
        answer_index=item["answer_index"],
    )
    assert len(qa.choices) == n_choices
    assert qa.answer_label in "ABCDEF"
    assert qa.wrong_label(rotate=3) in "ABCDEF"
