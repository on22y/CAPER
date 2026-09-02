# Experiment Asset Provenance

This development document records the mapping from the read-only experiment workspace to the
portable repository. Absolute source paths are retained here only for auditability.

| Canonical repository file | Source of truth | Evidence and treatment |
|---|---|---|
| `data/processed/dpo_train_498.jsonl` | `/home/user20250805/gayeon/HCLT2026/work/multimodel_dpo498_v1/data/dpo_preferences_frozen_498.jsonl` | Existing frozen cross-model artifact; copied byte-for-byte; manifest reports 498 unique IDs and SHA256 `5e5e...d5d1`. |
| `data/processed/heldout_51.jsonl` | `/home/user20250805/gayeon/HCLT2026/work/hcx_dpo900_mode_jury_v1/data/splits/final_heldout_source_51_final700.jsonl` | Input named by the successful Base-vs-DPO generation manifest; copied byte-for-byte; 51 IDs, 17 per domain, SHA256 `8c66...58f1`. |
| `configs/training/dpo_common.json` | `work/hcx_dpo501_train_v3/scripts/run_lora_dpo_checkpointed.py`, `metadata/train_status.json`, and `logs/train.log` | Portability config transcribed from the completed 375-step run and the installed TRL defaults. The historical run consumed 501 rows; the 498 set is the later frozen comparison subset. |
| `src/training/run_dpo.py` | Above successful HyperCLOVAX runner plus `work/multimodel_dpo498_v1/scripts/run_multimodel_dpo498.py` and the Qwen/Gemma FSDP feasibility worker | New portable orchestration only; DPO/LoRA values are preserved, Auto classes and architecture validation replace server-specific paths. No model was loaded during repository preparation. |
| `src/generation/run_base_vs_dpo_final51.py` | `work/hcx_base_vs_dpo501_final51_v1/scripts/run_base_vs_dpo501_final51.py` and successful Mistral generic adaptation | New portable orchestration; prompt, seed, strict JSON, retry, and decoding settings preserved. Model-specific official chat templates replace HyperCLOVAX-only encoding. |
| `prompts/rewrite/document_rewrite_instruction_v1.txt` | `work/hcx_base_vs_expanded_dpo_final60_v1/prompts/document_rewrite_instruction_v1.txt` | Existing repository copy matches the final generation prompt provenance; SHA256 `ff5528cc65e6f378e2f05c6fe4c05c7d7367509c731fe0e31c35721ccd63a798`. |
| `src/evaluation/canonical_automatic_metrics.py` | `work/hcx_dpo367_dataset_audit_v1/scripts/run_dpo367_automatic_metrics.py` | Existing portable copy of the canonical metric primitives; only repository-relative imports and optional model-path environment overrides were added. |
| `src/evaluation/metric_suite.py` | `work/hcx_dpo501_final_metrics_v1/scripts/run_final_metrics.py` | Metric definitions retained; canonical import repointed to the repository copy. |
| `src/evaluation/run_automatic_metrics.py` | `work/hcx_base_vs_dpo501_final51_eval_v1/scripts/run_automatic_metrics.py` | Final Base-vs-DPO aggregation retained. `run_comparison_metrics.py` supplies portable input/output routing. |

## Successful HyperCLOVAX reference

- Completed at 375/375 optimizer steps, epoch 3.0.
- Runtime: 9297.728 seconds; final train loss: 0.306577103972435; finite loss confirmed.
- Historical TRL input: 501 rows. The preflight labelled three overlength prompts as dropped, but
  the training log shows all 501 rows were added and tokenized. They were handled by TRL truncation.
- Frozen cross-model input: 498 rows, excluding those same three IDs. This pre-existing artifact is
  used for Qwen/Gemma comparability and was not regenerated in this repository-preparation task.
- Hardware: one model-parallel process across 2 x RTX 4090. Effective batch size was 4.
- No checkpoint resume was used for the successful v3 run; it started from Base + fresh LoRA.

## Canonical metric models

| Metric | Model |
|---|---|
| Entity F1 | `chunwoolee0/klue_ner_roberta_model` |
| BiNLI Support/Coverage/F1 | `joeddav/xlm-roberta-large-xnli` |
| Korean BERTScore P/R/F1 | `xlm-roberta-large` |

The original final metric manifest marked all three as locally prepared. Model weights are not
copied into Git.
