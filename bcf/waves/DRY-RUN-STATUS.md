# Status of the committed `dry-run-<wave>.txt` files after the R2 repricing

The 96 `dry-run-*.txt` files in this directory were produced on the cluster on
2026-09-07 at about 03:00, before ruling R2. **They are STALE with respect to the
manifests beside them** and are kept only as the record of that run, not as a check of
the current TSVs. Three things in them no longer describe what would be submitted:

1. every `--export` line carries `BCF_CONCURRENCY=1` and no `BCF_BATCH_INVARIANT`,
   where the current manifests carry `BCF_CONCURRENCY=32` and `BCF_BATCH_INVARIANT=1`;
2. every `BCF_EXPECTED_HOURS` is the sequential-client projection (7.585 h for an 8B
   cell at n 1,500) against the repriced 1.477 h;
3. the sbatch script path in them is `~/bcf/repo/bcf/serve_and_run.sbatch`, which the
   per-commit tree rule of DECISION-LOG 2026-09-07 03:58 has since made a forbidden
   target; `wave.sh` now resolves `~/bcf/repo-<short sha>`.

## Why they were not regenerated in this lane

`wave.sh --check-only` needs the cluster for the half of its work that matters most:
`squeue` for the live per-user card caps counting every campaign on the account, the
32-jobs-in-system limit, and `sbatch --test-only` on each command line. `ssh soc` was
unreachable for this entire lane (banner-exchange timeouts from 09:45 to past 10:10 on
2026-09-07; the root cause recorded at 08:36 is a `192.168.0/16` route via the home
gateway shadowing xlogin's addresses). Simulating an empty queue locally would produce
files that look like cluster output and are not, which is worse than none.

## What WAS checked locally, and how to re-check

`bcf/check_wave_manifests.py` runs the structural half: the serving constants on every
row, the pinned revision against the element 10 roster, the tensor-parallel size against
the cards per node, the curve cap against n per cell, the substrate and cue family
against the frozen lists, and each wave's card count against its pool's CONTRACT sweep
split. Result at the commit that carries this file: `local-manifest-check.txt`, 96 files,
216 rows, no structural problem. It is proven able to refuse:
`docs/a3f-proofs/MANIFEST_CHECK_EXIT_CODES.txt` records exit 1 when one row loses its
`BCF_BATCH_INVARIANT`.

When the cluster answers again, the live half is one loop:

```
ssh soc
cd ~/bcf/repo-<short sha of the planning commit>
for f in bcf/waves/*.tsv; do
  pool=$(basename "$f" .tsv); pool=${pool%-*}; pool=${pool%-single}; pool=${pool%-tp2}
  bash bcf/wave.sh --type sweep --gpu-type "$pool" --check-only "$f" \
    > "bcf/waves/dry-run-$(basename "$f" .tsv).txt" 2>&1
  echo "exit_code=$?" >> "bcf/waves/dry-run-$(basename "$f" .tsv).txt"
done
```

Read the refusals rather than counting them: the 36 a100-80 refusals in the stale files
were a live pool state (the alta campaign holding 3 of 4 cards), not a defect in the
manifests.
