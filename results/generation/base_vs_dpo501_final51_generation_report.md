# Base vs DPO501 Final51 Generation Report

## Dataset

- final frozen held-out: 51
- news / petition / sns: 17 / 17 / 17
- SHA256: 8c66122c61c756a732d24576440dff0b8ec546338d4016769d563bb8623058f1

## Method

Same generation methodology as the previous Base-vs-DPO experiment.
Only intended changes: final700 held-out manifest, DPO501 adapter, and new output root.
Method equivalence: PASS.

## Base

- valid: 51/51
- invalid: 0
- raw attempts: 58
- adapter loaded: False

## DPO501

- valid: 51/51
- invalid: 0
- raw attempts: 57
- adapter loaded: True
- adapter path: /home/user20250805/gayeon/HCLT2026/work/hcx_dpo501_train_v3/checkpoints/final_adapter
- old DPO adapter loaded: False

## Paired

- paired valid: 51/51
- missing IDs: []

## Failures

```json
[]
```
