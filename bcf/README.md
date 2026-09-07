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
