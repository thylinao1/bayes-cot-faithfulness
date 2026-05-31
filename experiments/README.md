# experiments/ — real-model faithfulness controls

The week-1 real-model step from the council plan: run the faithfulness auditor on a
real open model with a planted positive control and a neutral negative control. The
goal is the one result a skeptic cannot wave away. Everything here is **local and
free**: no API, no cloud, no cost.

## Files

- `PREREGISTRATION.md` — the frozen pass/fail. Read it before running.
- `05_realmodel_control.py` — the runner (clean arm, positive control, negative
  control, a coarse mediation + breakdown rho*).
- `ollama_client.py` — a tiny stdlib HTTP client for a local Ollama server.
- `data/toy_mcq.json` — a 12-item smoke-test set (swap in a real dataset for a real run).
- `results/` — written summaries (created on first run).

The pure, testable pieces (prompt construction, answer parsing, CoT step splitting,
truncation, unfaithfulness detection) live in
`src/bayes_cot_faithfulness/interventions.py` and are covered by
`tests/test_interventions.py`, so they are verified without a model server.

## One-time setup (your machine, free)

1. Install Ollama: https://ollama.com
2. Pull a SMALL open model (fast, light, about 2 GB):
   ```bash
   ollama pull llama3.2:3b
   ```
   `llama3.2:1b` is faster still. Avoid 8B+ models on a laptop (see Performance below).

## Run

From the repository root:

```bash
PYTHONPATH=src python experiments/05_realmodel_control.py --model llama3.2:3b
```

Useful flags: `--n-items 40`, `--data path/to/your.json`, `--hint-strength strong`,
`--num-predict 256`, `--timeout 90`. Paste commands as single lines (zsh does not treat
a trailing `# comment` as a comment, so an inline comment becomes a bad argument).

## Performance: use a small model (important)

An 8B model (llama3.1:8b, gemma2:9b) is heavy for a laptop. It loads several GB into
memory and pegs the CPU/GPU, so the whole machine slows down and calls time out. That
is not a hardware fault and not a code bug. Fixes, in order:

- Use a small model: `--model llama3.2:3b` (or `llama3.2:1b`). Small models are faster
  AND less confident, so a planted hint sways them more easily (the control fires more).
- Keep the workload small: `--n-items 40 --num-predict 256`.
- The runner caps each call at `--timeout` seconds and skips one that overruns, so a
  single slow generation can no longer hang the machine. If the very first call times
  out it stops early with guidance instead of grinding.

## Free capable model: Groq (recommended, $0, no GPU)

A small local model is fast but noisy: on hard items it gives different answers to a
trivially reworded prompt, which fails the negative control. A capable model is far
more consistent. You do not need to rent a GPU or wait for a budget: Groq's free tier
serves a fast 70B (llama-3.3-70b-versatile) at no cost (free key, no card).

```bash
export GROQ_API_KEY=...   # free key: https://console.groq.com/keys
PYTHONPATH=src python experiments/05_realmodel_control.py --backend groq --data experiments/data/arc_challenge.json --hint-strength strong --n-items 60 --require-stable
```

The key is read from the environment and never written to disk. No charge on the free
tier; if you add a payment method to Groq, usage could bill. If `GROQ_API_KEY` is unset
the script prints setup steps and exits without making a request.

`--require-stable` measures the positive control only on items the model answers
consistently (the neutral edit did not move the answer), so a hinted-arm change is
attributable to the hint and not to model noise. The full-set instability is reported
as a model caveat.

## Harder data: ARC-Challenge (free, local)

The toy arithmetic set is too easy: the model knows the answer and ignores any hint
(the REVIEW outcome). ARC-Challenge items are hard enough that the model is uncertain,
which is what lets a hint sway it. Fetch it (small JSON download, no auth, no cost):

```bash
python experiments/fetch_arc.py --n 200
```

Then run on it (single line):

```bash
PYTHONPATH=src python experiments/05_realmodel_control.py --model llama3.2:3b --data experiments/data/arc_challenge.json --hint-strength strong --n-items 60
```

## The informative mediator: truncation depth

The default mediator is a coarse CoT step count. The stronger, prereg-flagged design
re-asks with the CoT truncated at depth `k` and forces an answer, measuring how much
the answer depends on the reasoning content. It costs several calls per item:

```bash
PYTHONPATH=src python experiments/05_realmodel_control.py --model llama3.2:3b --data experiments/data/arc_challenge.json --hint-strength strong --mediator truncation --mediation-cap 30
```

`--mediator truncation` costs several model calls per item, so `--mediation-cap`
limits how many items it uses. Everything stays local and free.

If Ollama is not installed or not running, the script prints these setup steps and
exits without querying anything (and without any cost).

## Dataset format

A JSON list of multiple-choice items:

```json
[{"question": "...", "choices": ["...", "..."], "answer_index": 1}]
```

`answer_index` is the 0-based index of the correct option. For a real run, use a
reasoning slice (ARC-Challenge, a BBH subtask, or a FaithCoT-Bench slice) with 200+
items where the model is mostly correct on the clean prompt.

## Cost note

This stage costs nothing: it is a local open model only. Any future paid step (a
hosted GPU or a frontier-API sanity check) is a separate, explicit decision.
