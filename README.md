# HCLT 2026

본 저장소는 **한국어 문서의 사실 보존형 표현 재작성(fact-preserving expressive rewriting)** 실험을 재현하기 위한 코드, 프롬프트, 설정 및 주요 실험 artifact를 제공합니다.

## Pipeline

```text
900 Source Documents
    ↓
Conservative / Transformative Rewrite Generation
    ↓
888 Complete C/T Pairs
    ↓
3-Model LLM Jury
(Qwen3 / Granite / Ministral3)
Original + Swapped Evaluation
    ↓
Order-Stable Vote Aggregation
    ↓
501 DPO Preference Pairs
    ↓
LoRA-DPO Training
(498 Effective Samples)
    ↓
Frozen Held-out 51
(News 17 / Petition 17 / SNS 17)
    ↓
Base vs. DPO-Adapted Generation
    ↓
Automatic Metrics + GPT-4o Evaluation
````

## Models

### Base / Rewrite Model

* `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B`

### LLM Jury

| Judge      | Hugging Face Model ID                    | Revision                                   |
| ---------- | ---------------------------------------- | ------------------------------------------ |
| Qwen3      | `Qwen/Qwen3-8B`                          | `b968826d9c46dd6066d109eabc6255188de91218` |
| Granite    | `ibm-granite/granite-3.3-8b-instruct`    | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |
| Ministral3 | `mistralai/Ministral-3-8B-Instruct-2512` | `5b26027e7b19eeb4b7352e1fed3926375dd2cb4d` |

### API Evaluator

* `gpt-4o`

## Repository Structure

```text
.
├── src/          # 최종 실험 코드
├── prompts/      # Rewrite / Jury / Evaluation prompts
├── configs/      # Generation / DPO / Model configs
├── data/         # Source, C/T rewrite, Jury, preference, held-out data
├── results/      # Generation 및 evaluation 결과
└── FINAL_FILE_MANIFEST.json
```

## Main Experimental Stages

| Stage                                           | Main Script                                          |
| ----------------------------------------------- | ---------------------------------------------------- |
| C/T generation + Jury + Preference construction | `src/preference/run_dpo900_mode_jury_pipeline.py`    |
| Held-out selection                              | `src/data/create_final700_heldout51.py`              |
| DPO preprocessing                               | `src/training/prepare_dpo501.py`                     |
| LoRA-DPO training                               | `src/training/run_dpo501_train.py`                   |
| Base vs. DPO generation                         | `src/generation/run_base_vs_dpo501_final51.py`       |
| Automatic evaluation                            | `src/evaluation/run_automatic_metrics.py`            |
| GPT-4o evaluation                               | `src/evaluation/run_gpt4o_substantive_similarity.py` |

## Evaluation

### Factual / Semantic Preservation

* Entity F1
* Numeric / Date F1
* BiNLI Support / Coverage / F1
* Korean BERTScore Precision / Recall / F1

### Expression Duplicate / Novelty

* chrF++
* Morpheme ROUGE-L
* Extractive Coverage / Density
* Novel-2 / Novel-4

GPT-4o evaluation은 SOURCE 대비 **substantive expressive similarity**를 측정하며, 각 sample을 Original / Swapped 두 candidate order로 평가합니다.

## Reproducibility

최종 artifact의 provenance와 SHA256 checksum은 `FINAL_FILE_MANIFEST.json`에 기록되어 있습니다.

본 저장소에는 pretrained model weights, LoRA/checkpoint weights, Hugging Face cache 및 API credential을 포함하지 않습니다.

일부 코드는 최종 실험 당시의 실행 환경을 그대로 보존하고 있어 내부 absolute path를 포함할 수 있습니다. **추후 repository-relative path 또는 configurable path로 정리할 예정입니다.**