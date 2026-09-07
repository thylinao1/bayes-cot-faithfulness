#!/bin/bash
# Proof that tests/test_hold_filter.py can actually fail, not just pass by construction.
#
# Three mutations, each breaking one property the hold list depends on, each caught by a
# named test. The file is restored after every mutation so the repo is unchanged when this
# script exits, whatever its own exit code.
#
#   bash bcf/hold_filter_falsification.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
TARGET="${REPO}/bcf/hold_filter.py"
BACKUP="$(mktemp)"
cp "$TARGET" "$BACKUP"
restore() { cp "$BACKUP" "$TARGET"; }
trap restore EXIT

PY="$(command -v python3 || command -v python)"

run_tests() {  # $1 = -k expression
  ( cd "$REPO" && "$PY" -m pytest tests/test_hold_filter.py -k "$1" -q 2>&1 )
}

echo "=== bcf/hold_filter.py: falsification, three mutations against tests/test_hold_filter.py ==="

echo
echo "--- baseline: the unmutated file, full suite"
run_tests ""
echo "BASELINE_EXIT=$?"

echo
echo "--- mutation 1: the hold-set membership check never fires (every row is 'kept')"
python3 - "$TARGET" <<'PYEOF'
import sys
p = sys.argv[1]
text = open(p).read()
needle = "        if model in hold:\n"
assert needle in text, "mutation 1 target line not found"
text = text.replace(needle, "        if False:  # MUTATED: hold set never matches\n", 1)
open(p, "w").write(text)
PYEOF
run_tests "test_filter_wave_text_keeps_header_and_splits_rows_on_the_five_held_models or test_cli_filter_writes_the_partial_file_and_the_extra_json"
M1=$?
echo "MUTATION_1_EXIT=$M1 (expected nonzero: filtering that never holds anything should fail the split tests)"
restore

echo
echo "--- mutation 2: the malformed-id regex is skipped (every non-blank line is accepted)"
python3 - "$TARGET" <<'PYEOF'
import sys
p = sys.argv[1]
text = open(p).read()
needle = "        if not _ID_RE.match(body):\n"
assert needle in text, "mutation 2 target line not found"
text = text.replace(needle, "        if False:  # MUTATED: id shape never checked\n", 1)
open(p, "w").write(text)
PYEOF
run_tests "test_parse_hold_list_refuses_a_line_with_no_slash_and_names_the_line or test_parse_hold_list_refuses_a_line_with_two_slashes or test_cli_filter_refuses_a_malformed_hold_line_and_writes_nothing"
M2=$?
echo "MUTATION_2_EXIT=$M2 (expected nonzero: a malformed line that is silently accepted should fail the refusal tests)"
restore

echo
echo "--- mutation 3: the held-only rebuild stops re-checking the current hold list (everything collected is kept)"
python3 - "$TARGET" <<'PYEOF'
import sys
p = sys.argv[1]
text = open(p).read()
needle = 'def compose_wave_text(rows: list[str], hold: set[str], header_comment: list[str]) -> FilterResult:'
assert needle in text, "mutation 3 target signature not found"
old_body_marker = "    text = \"\\n\".join(header_comment + rows)\n    return filter_wave_text(text, hold)\n"
assert old_body_marker in text, "mutation 3 target body not found"
new_body = "    text = \"\\n\".join(header_comment + rows)\n    return filter_wave_text(text, set())  # MUTATED: current hold list ignored\n"
text = text.replace(old_body_marker, new_body, 1)
open(p, "w").write(text)
PYEOF
run_tests "test_compose_rebuild_with_nothing_lifted_yet_keeps_nothing or test_cli_compose_with_gpt_oss_still_held_clears_only_the_other_four"
M3=$?
echo "MUTATION_3_EXIT=$M3 (expected nonzero: a rebuild that ignores the live hold list should fail the still-held tests)"
restore

echo
echo "--- restored: the unmutated file, full suite again"
run_tests ""
FINAL=$?
echo "FINAL_EXIT=$FINAL (expected 0: the file on disk is exactly what it was before this script ran)"
