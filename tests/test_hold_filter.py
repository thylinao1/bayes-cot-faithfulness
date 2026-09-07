"""The hold list, unit tested with no cluster and no live wave file.

bcf/hold_filter.py is the part of the feeder hold mechanism with no squeue dependency:
parsing HOLD_MODELS.txt, splitting one wave file's rows into kept and held, and
recombining held rows from several waves and checking them again against the current
hold list. Every test here proves the property it claims by also exercising the case
where the property would fail: a bad hold-list line is rejected with the RIGHT line
number, an unfiltered wave keeps every row, a wave with every model held reports zero
kept, and a row still on the hold list stays out of a held-only rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bcf.hold_filter import (
    HoldListError,
    compose_wave_text,
    filter_wave_text,
    parse_hold_list,
    partial_path,
    read_hold_list,
)

WAVE_TEXT = """\
# wave a100-40-02  pool a100-40  8 card(s)
# columns: MODEL <tab> SUBSTRATE <tab> CUE <tab> KEY=VALUE ...
Qwen/Qwen3-8B\tarc_challenge\tprofessor\tBCF_REVISION=aaa
allenai/Olmo-3-7B-Think\tarc_challenge\tprofessor\tBCF_REVISION=bbb
deepseek-ai/DeepSeek-R1-0528-Qwen3-8B\tarc_challenge\tprofessor\tBCF_REVISION=ccc
deepseek-ai/DeepSeek-R1-Distill-Llama-8B\tarc_challenge\tprofessor\tBCF_REVISION=ddd
google/gemma-2-9b-it\tarc_challenge\tprofessor\tBCF_REVISION=eee
meta-llama/Llama-3.1-8B-Instruct\tarc_challenge\tprofessor\tBCF_REVISION=fff
microsoft/Phi-4-reasoning\tarc_challenge\tprofessor\tBCF_REVISION=ggg
openai/gpt-oss-20b\tarc_challenge\tprofessor\tBCF_REVISION=hhh
"""

FIVE_HELD = {
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "microsoft/Phi-4-reasoning",
    "openai/gpt-oss-20b",
}


# --- parse_hold_list -------------------------------------------------------------

def test_parse_hold_list_reads_one_id_per_line_in_order():
    text = "allenai/Olmo-3-7B-Think\nopenai/gpt-oss-20b\n"
    assert parse_hold_list(text) == ["allenai/Olmo-3-7B-Think", "openai/gpt-oss-20b"]


def test_parse_hold_list_skips_blank_lines_and_whole_line_comments():
    text = "\n# a comment\n\nopenai/gpt-oss-20b\n   \n"
    assert parse_hold_list(text) == ["openai/gpt-oss-20b"]


def test_parse_hold_list_strips_a_trailing_comment_after_the_id():
    text = "openai/gpt-oss-20b   # DECISION-LOG 2026-09-07 15:40, gpt-oss serving line\n"
    assert parse_hold_list(text) == ["openai/gpt-oss-20b"]


def test_parse_hold_list_refuses_a_line_with_no_slash_and_names_the_line():
    with pytest.raises(HoldListError) as exc:
        parse_hold_list("openai/gpt-oss-20b\ngpt-oss-20b\n")
    assert "line 2" in str(exc.value)
    assert "gpt-oss-20b" in str(exc.value)


def test_parse_hold_list_refuses_a_line_with_two_slashes():
    """The falsification for the id shape: a second slash is not a valid id either."""
    with pytest.raises(HoldListError) as exc:
        parse_hold_list("openai/gpt-oss/20b\n")
    assert "line 1" in str(exc.value)


def test_read_hold_list_of_a_missing_file_is_the_empty_list_not_an_error(tmp_path):
    assert read_hold_list(tmp_path / "does-not-exist.txt") == []


# --- filter_wave_text -------------------------------------------------------------

def test_filter_wave_text_keeps_header_and_splits_rows_on_the_five_held_models():
    result = filter_wave_text(WAVE_TEXT, FIVE_HELD)
    assert result.n_total == 8
    assert result.n_kept == 3
    assert result.n_held == 5
    assert list(result.kept_models) == [
        "Qwen/Qwen3-8B", "google/gemma-2-9b-it", "meta-llama/Llama-3.1-8B-Instruct",
    ]
    assert set(result.held_models) == FIVE_HELD
    # the two '#' lines survive verbatim, in their original position
    assert result.header[0].startswith("# wave a100-40-02")
    assert result.header[1].startswith("# columns:")


def test_filter_wave_text_with_an_empty_hold_set_keeps_every_row():
    """The falsification for the split above: nothing held means nothing filtered."""
    result = filter_wave_text(WAVE_TEXT, set())
    assert result.n_kept == 8
    assert result.n_held == 0
    assert result.render() == WAVE_TEXT


def test_filter_wave_text_with_every_model_held_keeps_nothing():
    all_models = {
        "Qwen/Qwen3-8B", "allenai/Olmo-3-7B-Think",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "google/gemma-2-9b-it",
        "meta-llama/Llama-3.1-8B-Instruct", "microsoft/Phi-4-reasoning",
        "openai/gpt-oss-20b",
    }
    result = filter_wave_text(WAVE_TEXT, all_models)
    assert result.n_kept == 0
    assert result.n_held == 8
    # render() still carries the header (a caller that DOES write it stays self-explaining);
    # what proves nothing survived is that no data row is left, only the two '#' lines.
    kept_lines = [ln for ln in result.render().splitlines() if not ln.startswith("#")]
    assert kept_lines == []


def test_filter_wave_text_only_reads_column_one_for_the_model():
    """A model id appearing in a KEY=VALUE field must not accidentally match the hold set."""
    text = "openai/gpt-oss-20b\tarc\tcue\tBCF_NOTE=allenai/Olmo-3-7B-Think\n"
    result = filter_wave_text(text, {"allenai/Olmo-3-7B-Think"})
    assert result.n_kept == 1
    assert result.n_held == 0


# --- compose_wave_text (the held-only rebuild) -------------------------------------

def test_compose_rebuild_lets_through_a_row_whose_model_left_the_hold_list():
    rows = [
        "allenai/Olmo-3-7B-Think\tarc_challenge\tprofessor\tBCF_REVISION=bbb",
        "openai/gpt-oss-20b\tarc_challenge\tprofessor\tBCF_REVISION=hhh",
    ]
    # element 9.4 lifted the thinking models; the gpt-oss serving-line ruling has not
    # lifted gpt-oss-20b yet, so only one of the two rows should come back kept.
    still_held = {"openai/gpt-oss-20b"}
    result = compose_wave_text(rows, still_held, ["# held-only rebuild"])
    assert result.n_kept == 1
    assert result.kept_models == ("allenai/Olmo-3-7B-Think",)
    assert result.held_models == ("openai/gpt-oss-20b",)


def test_compose_rebuild_with_nothing_lifted_yet_keeps_nothing():
    """The falsification: if the hold list still names every collected row, none clears."""
    rows = [
        "allenai/Olmo-3-7B-Think\tarc_challenge\tprofessor\tBCF_REVISION=bbb",
        "openai/gpt-oss-20b\tarc_challenge\tprofessor\tBCF_REVISION=hhh",
    ]
    result = compose_wave_text(rows, FIVE_HELD, ["# held-only rebuild"])
    assert result.n_kept == 0
    assert result.n_held == 2


# --- partial_path -------------------------------------------------------------

def test_partial_path_names_the_file_by_stem_and_kept_count():
    p = partial_path(Path("/x/bcf/waves"), "a100-40-02", 3)
    assert p == Path("/x/bcf/waves/partial/a100-40-02-3rows.tsv")


# --- the CLI, end to end, exercised the way wave_feeder.sh calls it -----------------

def test_cli_filter_writes_the_partial_file_and_the_extra_json(tmp_path):
    from bcf.hold_filter import main as hf_main

    wave = tmp_path / "a100-40-02.tsv"
    wave.write_text(WAVE_TEXT)
    hold = tmp_path / "HOLD_MODELS.txt"
    hold.write_text("\n".join(sorted(FIVE_HELD)) + "\n")
    waves_dir = tmp_path / "waves"
    waves_dir.mkdir()
    extra = tmp_path / "extra.json"

    rc = hf_main([
        "filter", str(wave), str(hold),
        "--waves-dir", str(waves_dir), "--extra-out", str(extra),
    ])
    assert rc == 0
    partial = waves_dir / "partial" / "a100-40-02-3rows.tsv"
    assert partial.exists()
    kept_text = partial.read_text()
    assert kept_text.count("\n") - 2 == 3  # two header lines plus three data rows
    extra_data = json.loads(extra.read_text())
    assert extra_data["status"] == "submitted"
    assert len(extra_data["submitted_models"]) == 3
    assert len(extra_data["held_models"]) == 5
    assert len(extra_data["held_rows"]) == 5
    assert extra_data["partial_file"] == "bcf/waves/partial/a100-40-02-3rows.tsv"


def test_cli_filter_with_an_empty_hold_file_writes_no_partial_file(tmp_path):
    """Proof (2)'s shape at the unit level: nothing held, nothing partial."""
    from bcf.hold_filter import main as hf_main

    wave = tmp_path / "a100-40-02.tsv"
    wave.write_text(WAVE_TEXT)
    hold = tmp_path / "HOLD_MODELS.txt"
    hold.write_text("")
    waves_dir = tmp_path / "waves"
    waves_dir.mkdir()
    extra = tmp_path / "extra.json"

    rc = hf_main([
        "filter", str(wave), str(hold),
        "--waves-dir", str(waves_dir), "--extra-out", str(extra),
    ])
    assert rc == 0
    assert not (waves_dir / "partial").exists()
    extra_data = json.loads(extra.read_text())
    assert extra_data["partial_file"] is None
    assert len(extra_data["submitted_models"]) == 8
    assert extra_data["held_rows"] == []


def test_cli_filter_refuses_a_malformed_hold_line_and_writes_nothing(tmp_path):
    """Proof (5) at the unit level."""
    from bcf.hold_filter import main as hf_main

    wave = tmp_path / "a100-40-02.tsv"
    wave.write_text(WAVE_TEXT)
    hold = tmp_path / "HOLD_MODELS.txt"
    hold.write_text("openai/gpt-oss-20b\ngpt-oss-20b\n")
    waves_dir = tmp_path / "waves"
    waves_dir.mkdir()
    extra = tmp_path / "extra.json"

    rc = hf_main([
        "filter", str(wave), str(hold),
        "--waves-dir", str(waves_dir), "--extra-out", str(extra),
    ])
    assert rc == 13
    assert not extra.exists()
    assert not (waves_dir / "partial").exists()


def test_cli_collect_gathers_held_rows_for_the_named_pool_and_type_only(tmp_path):
    from bcf.hold_filter import main as hf_main

    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "waves": {
            "a100-40-02.tsv": {
                "gpu_type": "a100-40", "job_type": "sweep",
                "held_rows": ["allenai/Olmo-3-7B-Think\tarc\tprofessor\tBCF_REVISION=bbb"],
            },
            "a100-40-03.tsv": {
                "gpu_type": "a100-40", "job_type": "sweep",
                "held_rows": ["openai/gpt-oss-20b\tarc\tprofessor\tBCF_REVISION=hhh"],
            },
            "a100-80-01.tsv": {
                # a different pool: must not be collected
                "gpu_type": "a100-80", "job_type": "sweep",
                "held_rows": ["openai/gpt-oss-20b\taqua_rat\tprofessor\tBCF_REVISION=zzz"],
            },
            "a100-40-04.tsv": {
                # nothing held here: contributes no rows and is not a source
                "gpu_type": "a100-40", "job_type": "sweep",
                "held_rows": [],
            },
        }
    }))
    rows_out = tmp_path / "rows.txt"
    sources_out = tmp_path / "sources.json"

    rc = hf_main([
        "collect", str(state), "a100-40", "sweep",
        "--rows-out", str(rows_out), "--sources-out", str(sources_out),
    ])
    assert rc == 0
    rows = rows_out.read_text().splitlines()
    assert len(rows) == 2
    assert any(r.startswith("allenai/Olmo-3-7B-Think") for r in rows)
    assert any(r.startswith("openai/gpt-oss-20b") for r in rows)
    assert "aqua_rat" not in rows_out.read_text()
    sources = json.loads(sources_out.read_text())["source_waves"]
    assert sources == ["a100-40-02.tsv", "a100-40-03.tsv"]


def test_cli_collect_of_a_missing_state_file_is_zero_rows_not_an_error(tmp_path):
    from bcf.hold_filter import main as hf_main

    rows_out = tmp_path / "rows.txt"
    sources_out = tmp_path / "sources.json"
    rc = hf_main([
        "collect", str(tmp_path / "does-not-exist.json"), "a100-40", "sweep",
        "--rows-out", str(rows_out), "--sources-out", str(sources_out),
    ])
    assert rc == 0
    assert json.loads(sources_out.read_text())["source_waves"] == []


def test_cli_collect_of_a_torn_state_file_refuses_rather_than_guessing(tmp_path):
    from bcf.hold_filter import main as hf_main

    state = tmp_path / "state.json"
    state.write_text("{not valid json")
    rc = hf_main([
        "collect", str(state), "a100-40", "sweep",
        "--rows-out", str(tmp_path / "rows.txt"), "--sources-out", str(tmp_path / "s.json"),
    ])
    assert rc == 9


def test_cli_compose_builds_the_five_row_held_only_wave(tmp_path):
    """Proof (3) at the unit level: the exact five-model union a100-40-02 held."""
    from bcf.hold_filter import main as hf_main

    rows_file = tmp_path / "held_rows.txt"
    rows_file.write_text("\n".join(
        f"{m}\tarc_challenge\tprofessor\tBCF_REVISION=x" for m in sorted(FIVE_HELD)
    ) + "\n")
    hold = tmp_path / "HOLD_MODELS.txt"
    hold.write_text("")  # every hold lifted: the whole union should now clear
    waves_dir = tmp_path / "waves"
    waves_dir.mkdir()
    extra = tmp_path / "extra.json"

    rc = hf_main([
        "compose", str(rows_file), str(hold),
        "--waves-dir", str(waves_dir), "--name", "held-a100-40-sweep-01",
        "--extra-out", str(extra),
    ])
    assert rc == 0
    partial = waves_dir / "partial" / "held-a100-40-sweep-01-5rows.tsv"
    assert partial.exists()
    data_rows = [ln for ln in partial.read_text().splitlines() if not ln.startswith("#")]
    assert len(data_rows) == 5
    assert {ln.split("\t", 1)[0] for ln in data_rows} == FIVE_HELD


def test_cli_compose_with_gpt_oss_still_held_clears_only_the_other_four(tmp_path):
    from bcf.hold_filter import main as hf_main

    rows_file = tmp_path / "held_rows.txt"
    rows_file.write_text("\n".join(
        f"{m}\tarc_challenge\tprofessor\tBCF_REVISION=x" for m in sorted(FIVE_HELD)
    ) + "\n")
    hold = tmp_path / "HOLD_MODELS.txt"
    hold.write_text("openai/gpt-oss-20b   # still held, serving line unresolved\n")
    waves_dir = tmp_path / "waves"
    waves_dir.mkdir()
    extra = tmp_path / "extra.json"

    rc = hf_main([
        "compose", str(rows_file), str(hold),
        "--waves-dir", str(waves_dir), "--name", "held-a100-40-sweep-01",
        "--extra-out", str(extra),
    ])
    assert rc == 0
    extra_data = json.loads(extra.read_text())
    assert extra_data["held_models"] == ["openai/gpt-oss-20b"]
    assert len(extra_data["submitted_models"]) == 4
