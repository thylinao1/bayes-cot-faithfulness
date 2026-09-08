# Row-splitting shared by bcf/wave.sh and its tests.
#
# Kept in its own file, and kept free of bash4-only syntax (no `declare -A`, no
# `mapfile`), so it can be sourced and exercised under the Mac's system bash (3.2) as
# well as bash 5. bcf/wave.sh itself needs bash 4+ for its associative arrays and
# `mapfile`, so this is the only part of the row-parsing that a bash-3.2 test can run
# directly.
#
# Split a row on TABS only. `for f in $(... | tr '\t' '\n')` also splits on spaces, which
# silently shatters a value like `BCF_EXTRA_VLLM=--max-num-seqs 64` into two fields and
# exports a variable named `64`.
row_fields() { printf '%s\n' "$1" | tr '\t' '\n'; }
