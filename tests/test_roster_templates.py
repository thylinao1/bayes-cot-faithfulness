"""Offline tests for the pure classifier in experiments/roster_templates.py.

Three small, hand-written chat templates stand in for the three shapes this lane's
18-row read actually found: a documented switch (Qwen3's `enable_thinking`, class 1),
an unconditionally opened block with no switch (Olmo/R1-Distill's `<think>`, class 2),
and a plain instruct template with no reasoning token anywhere (Llama/Gemma, class 3).
A fourth template stands in for the class-4 "other" catch-all this lane needed for two
real rows (DeepSeek-R1-0528-Qwen3-8B, Phi-4-reasoning): a reasoning marker that lives
OUTSIDE the generation-prompt tail, which is exactly what makes those two rows invisible
to a classifier that only reads the tail. No network: `fetch_row` is not exercised here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "roster_templates_under_test", REPO / "experiments" / "roster_templates.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()

SWITCH_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "    {{- '<|im_start|>assistant\\n' }}"
    "    {%- if enable_thinking is defined and enable_thinking is false %}"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}"
    "    {%- endif %}"
    "{%- endif %}"
)

OPENED_BLOCK_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n<think>' }}"
    "{% endif %}"
)

PLAIN_TEMPLATE = (
    "{{ bos_token }}"
    "{%- for message in messages %}"
    "{{- '<start_of_turn>' + message.role + '\\n' + message.content | trim + '<end_of_turn>\\n' }}"
    "{%- endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
)

# A reasoning marker OUTSIDE the tail: the generation-prompt block itself is bare,
# same shape as PLAIN_TEMPLATE's tail, but the message loop rewrites a prior
# assistant turn by splitting off a closing think tag, which only a reasoning
# model's template would ever need to do.
OTHER_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'assistant' %}"
    "{%- set content = message.content %}"
    "{%- if '</think>' in content %}{%- set content = content.split('</think>')[-1] %}{%- endif %}"
    "{{- '<|Assistant|>' + content }}"
    "{%- else %}"
    "{{- '<|User|>' + message.content }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{% if add_generation_prompt %}{{ '<|Assistant|>' }}{% endif %}"
)


def test_switch_template_is_class_1_with_the_kwarg_and_default_named():
    result = mod.classify_chat_template(SWITCH_TEMPLATE)
    assert result["class"] == 1
    assert result["switch_kwarg"] == "enable_thinking"
    # the guard fires on the FALSE state, so the undefined (default) state is ON
    assert result["switch_default"] is True
    assert result["opened_tag"] is None


def test_opened_block_template_is_class_2_with_the_tag_named():
    result = mod.classify_chat_template(OPENED_BLOCK_TEMPLATE)
    assert result["class"] == 2
    assert result["switch_kwarg"] is None
    assert result["opened_tag"] == "<think>"


def test_plain_template_is_class_3():
    result = mod.classify_chat_template(PLAIN_TEMPLATE)
    assert result["class"] == 3
    assert result["switch_kwarg"] is None
    assert result["opened_tag"] is None


def test_other_template_with_a_marker_outside_the_tail_is_class_4_not_class_3():
    """The two real rows this class exists for (DeepSeek-R1-0528-Qwen3-8B,
    Phi-4-reasoning) have a bare generation-prompt tail; what makes them
    reasoning templates is elsewhere in the file. A tail-only classifier would
    wrongly call this class 3."""
    result = mod.classify_chat_template(OTHER_TEMPLATE)
    assert result["class"] == 4
    assert result["switch_kwarg"] is None
    assert result["opened_tag"] is None


def test_empty_template_text_classifies_as_no_template_rather_than_crashing():
    result = mod.classify_chat_template("")
    assert result["class"] is None


def test_assistant_rewrite_detected_on_the_other_template_and_absent_elsewhere():
    assert mod.detect_assistant_rewrite(OTHER_TEMPLATE)["rewrites"] is True
    assert mod.detect_assistant_rewrite(SWITCH_TEMPLATE)["rewrites"] is False
    assert mod.detect_assistant_rewrite(PLAIN_TEMPLATE)["rewrites"] is False


def test_switch_default_reversed_polarity_is_read_correctly():
    """A guard of the opposite polarity (`enable_thinking is true`) means the
    undefined/default state is OFF, the mirror image of the real Qwen3 guard."""
    reversed_polarity = SWITCH_TEMPLATE.replace(
        "enable_thinking is false", "enable_thinking is true"
    )
    result = mod.classify_chat_template(reversed_polarity)
    assert result["class"] == 1
    assert result["switch_default"] is False


def test_roster_has_eighteen_rows_with_unique_indices_one_through_eighteen():
    assert len(mod.ROSTER) == 18
    assert sorted(r["index"] for r in mod.ROSTER) == list(range(1, 19))
