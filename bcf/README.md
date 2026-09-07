# bcf/ - cluster scripts for the Phase-2 sweep

Everything here runs on the NUS SoC Compute Cluster (`ssh soc`), on free GPU hours.
Nothing here is paid, and nothing here calls an external API: the models are served by
vLLM on an allocated card and reached over `http://127.0.0.1:<port>/v1`, on a port
picked free per job (MIG slices of one node share the host loopback, so a fixed port
is a shared port).

Read `~/Developer/NUS-COMPUTE.md` sections 1.3 to 1.5 and 1.9 first. The two facts that
shape every script in this directory:

- **`MaxSubmitJobs=32` rejects the 33rd `sbatch` at submit time.** It is not queued, it
  is dropped. Array elements count individually.
- **The per-user `a100-80` cap of 8 counts every campaign on the account.** Another
  project's jobs consume the same budget, and going over does not error, it just makes
  your jobs queue invisibly behind theirs.

## Files

| File | What it is |
|---|---|
| `env.sh` | Sourced by every sbatch script. Activates conda env `bcf`, puts `HF_HOME` on node-local scratch, and defines `bcf_assert_devices`, `bcf_revision`, `bcf_pick_port` and `bcf_wait_server_ready`. |
| `serve_ready.py` | Readiness WITH ownership: the port must answer with our model list, from our live server pid, on a socket held by that pid or a descendant. Two jobs on one node shared a server on 2026-09-07 because the old probe asked only whether something answered. |
| `serve_and_run.sbatch` | One model x substrate x cue family. Serves the model, runs the forced-logprob check, runs the arms with `--resume`, measures throughput, writes `exit_code.txt`. |
| `tp2_serving_test.sbatch` | The tensor-parallel-2 serving test on two `a100-80`, for the 70B dense serving line. |
| `wave.sh` | The submit gate. Refuses a wave that would breach either cap. The only thing that should call `sbatch`. |
| `logprob_check.py` | Forced-answer-logprob unit check against a live server. |
| `throughput.py` | Generations per second per arm, from the run's own `run.log` and `requests.jsonl`. |
| `cells/*.tsv` | Wave definitions: one cell per line. |

## The environment

Built once by `~/bcf/env-setup.sbatch` (job 825198, `EXIT_CODE=0`, 2026-09-06) on a GPU
node, because **torch cannot be imported on the login node** (`libtorch_cpu.so: failed
to map segment`). Anything that imports vllm or torch must run inside a job.

    vllm 0.28.0   nnsight 0.7.0   numpyro 0.21.0   torch 2.13.0+cu130

## Running a wave

    rsync -avP --delete ~/Developer/bcf-sweep/ soc:~/bcf/repo/ \
      --exclude .git --exclude .venv --exclude __pycache__

    ssh soc 'bash ~/bcf/repo/bcf/wave.sh --type sweep --dry-run ~/bcf/repo/bcf/cells/<wave>.tsv'
    ssh soc 'bash ~/bcf/repo/bcf/wave.sh --type sweep           ~/bcf/repo/bcf/cells/<wave>.tsv'

`wave.sh` prints both denominators before it decides:

    [wave] jobs in system: 6/32; this wave adds 1 -> 7/32
    [wave] a100-80 cards in use, ALL campaigns: 4/8
    [wave] a100-80 cards in use, bcf-sweep: 0/4 (CONTRACT.md split)

A cells file is tab-separated, `#` comments allowed:

    MODEL <tab> SUBSTRATE <tab> CUE [<tab> BCF_KEY=VALUE ...]

`BCF_TP=2` makes `wave.sh` submit `--nodes=1 --gpus-per-node=a100-80:2` and counts two
cards against the budget instead of one.

## What `serve_and_run.sbatch` refuses to do

Each of these exits before spending anything, with a distinct code in `exit_code.txt`:

| Exit | Refusal |
|---|---|
| 2 | the results path does not end in `/<substrate>/<cue>` (CONTRACT layout) |
| 3 | `nvidia-smi -L` device count != the requested tensor-parallel size, **checked before the weights download** |
| 4 | the HF revision sha could not be resolved (a floating `main` is not reproducible) |
| 5 / 6 | the vLLM server died during startup / never answered |
| 7 | the forced-answer-logprob check failed |
| 13 | the port answers, but not from this job's server: a foreign model list, or a listener that is not the vLLM process this job started (`serve_ready.py`) |

Exit 3 is the one that matters most in practice. A 70B started on one card downloads
140 GB to scratch and then OOMs; the assertion turns that into a five-second exit.

## `BCF_REASONING_MODE` and the two-path gate (ruling R12)

Four roster rows document no reasoning-mode switch and their chat templates open a
reasoning block that the frozen 320-token budget cuts before an answer appears
(`docs/WAVE1-AUDIT.md`). R12 defines the switch, for those rows, as closing the block the
template opens, and `serve_and_run.sbatch` carries it as one variable:

| `BCF_REASONING_MODE` | What the cell does | Status |
|---|---|---|
| `default` (the default) | today's behaviour, byte for byte: `/chat/completions`, `num_predict` 320 full and 24 forced | what every other row runs |
| `off` | the prompt is rendered through the model's own template (`/tokenize` plus `/detokenize`), the reasoning block is CLOSED, generation moves to `/completions`, every element 15 constant unchanged | the CELL OF RECORD for those rows, once the gate below has passed |
| `on` | the ordinary chat path, `num_predict` 4096 for FULL generations on that row only (forced stays 24), the answer read after the closing think tag | the EXPLORATORY additive arm; reported beside the `off` cell per model and never pooled with it |

Every generation record, the cell summary and `run_meta.json` carry `reasoning_mode`,
`reasoning_path` (`chat` or `completions`), `reasoning_block_closed` and
`num_predict_full`; the runner refuses to write a checkpoint if any of them is missing.
`run_meta.json` also gets the rendered prompt tail and the count of forced continuations
that reopened a block anyway. An unknown mode exits 14 before the weights download.

    sbatch --job-name=bcf-sweep-olmo-3-7b-think-arc-stated-hint \
      --export=ALL,BCF_MODEL=allenai/Olmo-3-7B-Think,BCF_SUBSTRATE=arc_challenge,\
    BCF_CUE=stated-hint,BCF_REASONING_MODE=off ~/bcf/repo/bcf/serve_and_run.sbatch

Raise `BCF_MAX_LEN` to 16384 with `BCF_REASONING_MODE=on`; a 4096-token generation
against the 8192 default leaves little room for the prompt, and the job warns rather than
refuses.

**The gate that precedes an `off` cell of record.** R12(2): the request path change has to
be shown not to be a serving-mode change. `bcf/gate_twopath.sbatch` runs 30 ARC items on
`Qwen/Qwen3-8B` at the pinned revision under `VLLM_BATCH_INVARIANT=1`, sending each item
through `/chat/completions` and through the rendered `/completions` path with `default`
semantics (nothing closed, nothing appended). PASS is 30/30 byte-identical completions, a
max absolute letter-logprob difference of exactly 0.0, and a clean rendered-prompt round
trip; anything else exits 10 and the path change needs its own preflight line.

    ssh soc 'sbatch --export=ALL,BCF_REPO=$HOME/bcf/repo-rmode,\
    BCF_ENV_SH=$HOME/bcf/repo-rmode/bcf/env.sh $HOME/bcf/repo-rmode/bcf/gate_twopath.sbatch'

The determinism preflight and the forced-logprob check inside `serve_and_run.sbatch` stay
on the CHAT path whatever the mode (they measure the server), and `run_meta.json` records
that as `reasoning_preflight_path`. This gate is what certifies that the two paths render
the same prompt.

## Throughput is measured, not estimated

`OpenAIClient` appends one line per HTTP call to `$BCF_REQUEST_LOG`
(`<out_dir>/requests.jsonl`), and the sbatch script timestamps every line of the
runner's stdout into `run.log`. `throughput.py` joins the two: each call is assigned to
the arm whose interval contains its start time, and the rate is completed calls over
that arm's wall-clock seconds. Every rate is printed with its numerator and denominator.
An arm with no calls prints `n=0`, never a rate.

The result goes into `CONTRACT.md` under "Measured throughput (Phase 1)" and replaces
the first-order compute estimate as the budget of record.

## Etiquette (hard-learned; NUS-COMPUTE.md 1.9)

- **Never `scancel --me`, `scancel -u $USER`, or any wildcard cancel.** One session's
  teardown once destroyed another project's 13 GPU-hours mid-run. Enumerate with
  `squeue --me -o "%i %j"`, read the names, cancel explicit IDs. `serve_and_run.sbatch`
  kills only its own recorded server PID for the same reason.
- Job names carry the `bcf-<type>-` prefix so `squeue --me` shows which campaign owns
  what, and so `wave.sh` can count this campaign's cards separately from another's.
- Announce a large download before making it, and check `du -sh ~/` while you are there.
  Weights go to `/tmp/$SLURM_JOB_ID/hf` and die with the job; only results reach `$HOME`.
