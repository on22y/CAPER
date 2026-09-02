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
(historical run: 501 TRL rows / cross-model frozen set: 498 rows)
    ↓
Frozen Held-out 51
(News 17 / Petition 17 / SNS 17)
    ↓
Base vs. DPO-Adapted Generation
    ↓
Automatic Metrics + GPT-4o Evaluation
```

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

Qwen3-14B 및 Gemma-3-12B-it 비교 실험은 `docs/multimodel_comparison.md`의 clone, 환경 설치, 모델 준비, DPO, held-out 생성, 자동 평가 순서를 사용합니다. 원본 실험 자산과 portable 파일의 대응 관계는 `docs/experiment_provenance.md`에 기록되어 있습니다.

신규 서버에서는 아래 순서로 환경을 구성합니다. 모델 weight와 adapter weight는 저장소에 포함되어 있지 않으므로 `models/` 아래에 별도로 준비해야 합니다.

```bash
git clone https://github.com/on22y/HCLT_2026.git
cd HCLT_2026
conda create -n hclt python=3.10 -y
conda activate hclt
pip install -r requirements.txt
```

기본 경로는 repository root 기준입니다. 필요한 경우 다음 환경변수로 외부 checkpoint 위치를 지정할 수 있습니다.

```bash
export HCLT_BASE_MODEL=/path/to/HyperCLOVAX-SEED-Think-14B
export HCLT_DPO_ADAPTER=/path/to/dpo501_final_adapter
export OPENAI_API_KEY=...
```

최종 3-model LLM jury 설정은 `configs/evaluator_models.json`에 있으며 Qwen3 / Granite / Ministral3 기준으로 고정되어 있습니다. 일부 legacy/pilot helper script는 provenance 보존을 위해 과거 `work/hcx_*` 실험명 형태의 출력 경로를 유지할 수 있지만, 최종 제출 artifact와 주요 config는 repository-relative path를 사용합니다.