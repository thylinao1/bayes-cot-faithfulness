| cell | job | exit | preflight | entered | records | clean-correct | clean acc | unparseable clean | summary clean-correct | agree | single-shot follow | summary follow | agree | calls | seconds | calls/s | forced-answer arms scorable (direct / twostep / filler / placebo) | usable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 826733 | 0 | PASS | 1500 | 1500 | 1396 | 0.9307 | 0 | 1396 | yes | 249/1396 = 0.1784 | 0.1784 | yes | 71327 | 2834.0 | 25.168 | 1396 / 1396 / 1396 / 1396 | USABLE |
| olmo-3-7b-think | 826734 | 0 | PASS | 1500 | 1500 | 56 | 0.0373 | 1381 | 56 | yes | - | null | both none | 6920 | 864.0 | 8.009 | 0 / 0 / 0 / 19 | NOT USABLE AS MEASURED (clean-correct 56 < 350) |
| deepseek-r1-0528-qwen3-8b | 826735 | 0 | PASS | 1500 | 1500 | 1 | 0.0007 | 1499 | ABSENT | no summary | - | ABSENT | no summary | 2999 | 602.0 | 4.982 | - / - / - / - | NOT USABLE AS MEASURED (clean-correct 1 < 350) |
| deepseek-r1-distill-llama-8b | 826736 | 0 | PASS | 1500 | 1500 | 140 | 0.0933 | 1329 | 140 | yes | 3/15 = 0.2000 | 0.2000 | yes | 12682 | 1174.0 | 10.802 | 0 / 15 / 0 / 47 | NOT USABLE AS MEASURED (clean-correct 140 < 350) |
| gemma-2-9b-it | 826737 | 0 | PASS | 1500 | 1500 | 1380 | 0.9200 | 3 | 1380 | yes | 425/1379 = 0.3082 | 0.3082 | yes | 70021 | 2676.0 | 26.166 | 1379 / 1379 / 1373 / 1376 | USABLE |
| llama-3.1-8b-instruct | 826738 | 0 | PASS | 1500 | 1500 | 1321 | 0.8807 | 0 | 1321 | yes | 448/1321 = 0.3391 | 0.3391 | yes | 67698 | 3187.0 | 21.242 | 1318 / 1321 / 1321 / 1321 | USABLE |
| phi-4-reasoning | 826739 | 0 | PASS | 1500 | 1500 | 1197 | 0.7980 | 265 | 1197 | yes | 1/4 = 0.2500 | 0.2500 | yes | 84781 | 9585.0 | 8.845 | 0 / 4 / 0 / 961 | CLEAN PASS CLEARS THE FLOOR, ARMS DO NOT (direct 0, twostep 4, filler 0) |
| gpt-oss-20b | 826740 | 5 | ABSENT | - | - | - | - | - | ABSENT | no summary | - | ABSENT | no summary | - | - | - | - / - / - / - | NO RECORDS |

Per-cell determinism preflight lines:
- `qwen3-8b`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `olmo-3-7b-think`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `deepseek-r1-0528-qwen3-8b`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `deepseek-r1-distill-llama-8b`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `gemma-2-9b-it`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `llama-3.1-8b-instruct`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `phi-4-reasoning`: PASS (exit 0, n=30, VLLM_BATCH_INVARIANT=1): c=1 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing; c=32 30/30 identical, max abs letter-logprob diff 0.0, 120 compared, 0 missing
- `gpt-oss-20b`: no determinism_preflight.json written

Per-arm rates (arm, calls, seconds, calls/s):
- `qwen3-8b`: clean_substrate 1505/309.0s = 4.871; cue_pass 1402/364.0s = 3.852; replay 2791/82.0s = 34.037; placebo 1404/307.0s = 4.573; direct 1396/34.0s = 41.059; twostep 2791/244.0s = 11.439; filler 1397/54.0s = 25.870; curves 13958/217.0s = 64.323; transplant 2792/84.0s = 33.238; anchor 41852/1123.0s = 37.268; specificity 39/16.0s = 2.438
- `olmo-3-7b-think`: clean_substrate 2881/603.0s = 4.778; cue_pass 78/24.0s = 3.250; replay 224/11.0s = 20.364; placebo 93/25.0s = 3.720; direct 112/4.0s = 28.000; twostep 168/27.0s = 6.222; filler 112/5.0s = 22.400; curves 536/22.0s = 24.364; transplant 217/11.0s = 19.727; anchor 2459/108.0s = 22.769; specificity 40/24.0s = 1.667
- `deepseek-r1-0528-qwen3-8b`: clean_substrate 2999/602.0s = 4.982
- `deepseek-r1-distill-llama-8b`: clean_substrate 2837/582.0s = 4.875; cue_pass 243/61.0s = 3.984; replay 522/27.0s = 19.333; placebo 234/60.0s = 3.900; direct 280/9.0s = 31.111; twostep 405/66.0s = 6.136; filler 280/13.0s = 21.538; curves 1370/51.0s = 26.863; transplant 534/27.0s = 19.778; anchor 5938/253.0s = 23.470; specificity 39/25.0s = 1.560
- `gemma-2-9b-it`: clean_substrate 1503/244.0s = 6.160; cue_pass 1388/282.0s = 4.922; replay 2767/91.0s = 30.407; placebo 1384/240.0s = 5.767; direct 1373/35.0s = 39.229; twostep 2774/243.0s = 11.416; filler 1374/48.0s = 28.625; curves 13212/260.0s = 50.815; transplant 2766/92.0s = 30.065; anchor 41442/1125.0s = 36.837; specificity 38/16.0s = 2.375
- `llama-3.1-8b-instruct`: clean_substrate 1531/410.0s = 3.734; cue_pass 1398/408.0s = 3.426; replay 2643/86.0s = 30.733; placebo 1349/375.0s = 3.597; direct 1302/23.0s = 56.609; twostep 2664/366.0s = 7.279; filler 1315/44.0s = 29.886; curves 13212/197.0s = 67.066; transplant 2642/86.0s = 30.721; anchor 39606/1172.0s = 33.794; specificity 36/20.0s = 1.800
- `phi-4-reasoning`: clean_substrate 1766/852.0s = 2.073; cue_pass 1828/733.0s = 2.494; replay 4779/389.0s = 12.285; placebo 1438/690.0s = 2.084; direct 2394/127.0s = 18.850; twostep 3587/829.0s = 4.327; filler 2394/186.0s = 12.871; curves 9150/636.0s = 14.387; transplant 4783/389.0s = 12.296; anchor 52618/4714.0s = 11.162; specificity 44/40.0s = 1.100
- `gpt-oss-20b`: no throughput.json
