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
2. Pull a small open model (a few GB download, runs on-device):
   ```bash
   ollama pull llama3.1:8b      # or: ollama pull gemma2:9b
   ```

## Run

From the repository root:

```bash
PYTHONPATH=src python experiments/05_realmodel_control.py --model llama3.1:8b
```

Useful flags: `--n-items 200`, `--data path/to/your.json`, `--model gemma2:9b`,
`--hint-strength strong`.

If the run returns **REVIEW** because the model ignored the hint (as the toy run
did), re-run with a firmer cue and/or a harder set:

```bash
PYTHONPATH=src python experiments/05_realmodel_control.py --model llama3.1:8b --hint-strength strong
```

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
