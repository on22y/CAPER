# CAPER

**CAPER: Copyright-Aware Preference Learning for Expressively Independent Document Rewriting**

Gayeon Lee, Uk Jang, Jihie Kim
Dongguk University

**HCLT 2026 | The 38th Annual Conference on Human & Cognitive Language Technology**

[Official Program & Proceedings](https://sites.google.com/view/hclt2026/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8) | [Conference Website](https://sites.google.com/view/hclt2026)

## Overview

CAPER는 **사실과 의미를 보존하면서 원문 표현에 대한 의존도를 낮추기 위한 선호 학습 기반 문서 재작성 프레임워크**입니다.

동일한 원문에서 Conservative / Transformative 후보를 생성하고, 다중 LLM의 순서 일관 평가를 통해 선호 쌍을 구축한 뒤 Direct Preference Optimization(DPO)을 수행합니다.

본 저장소는 HCLT 2026 논문의 실험 재현을 위한 코드, 프롬프트, 설정 및 주요 실험 산출물(artifacts)을 제공합니다.

## Pipeline

```text
900 Source Documents
(News / Petition / SNS Reviews)
    ↓
Conservative / Transformative Generation
    ↓
888 Valid C/T Pairs
    ↓
3-Model LLM Jury
(Qwen3 / Granite / Ministral-3)
    ↓
Order-Consistent Preference Aggregation
    ↓
501 Final Preference Pairs
    ↓
LoRA-DPO Training
    ↓
Base vs. DPO Generation
    ↓
Automatic Metrics + GPT-4o Evaluation
```

평가에는 네 기반 모델에서 공통으로 확보된 **50개 held-out 문서**를 사용했습니다. HyperCLOVAX 인간 평가에는 초기 51개 held-out 문서를 사용했습니다.

## Models

**Primary Backbone**
- `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B`

**Additional Backbones**
- `Qwen3-14B`
- `Gemma-3-12B-IT`
- `Mistral-Nemo-Instruct`

**LLM Jury**

| Judge | Hugging Face Model ID |
|---|---|
| Qwen3 | `Qwen/Qwen3-8B` |
| Granite | `ibm-granite/granite-3.3-8b-instruct` |
| Ministral-3 | `mistralai/Ministral-3-8B-Instruct-2512` |

**API Evaluator:** `gpt-4o`

모델별 revision 및 평가 설정은 `configs/evaluator_models.json`에서 확인할 수 있습니다.

## Repository Structure

```text
.
├── src/            # Experimental source code
├── prompts/        # Rewrite / Jury / Evaluation prompts
├── configs/        # Generation / DPO / Model configurations
├── data/           # Source, rewrite, preference, held-out data
├── results/        # Generation and evaluation results
├── docs/           # Reproduction and provenance documentation
└── FINAL_FILE_MANIFEST.json
```

### Main Experimental Scripts

| Stage | Script |
|---|---|
| C/T Generation + Jury + Preference | `src/preference/run_dpo900_mode_jury_pipeline.py` |
| Held-out Selection | `src/data/create_final700_heldout51.py` |
| DPO Preprocessing | `src/training/prepare_dpo501.py` |
| LoRA-DPO Training | `src/training/run_dpo501_train.py` |
| Base vs. DPO Generation | `src/generation/run_base_vs_dpo501_final51.py` |
| Automatic Evaluation | `src/evaluation/run_automatic_metrics.py` |
| GPT-4o Evaluation | `src/evaluation/run_gpt4o_substantive_similarity.py` |

## Evaluation

CAPER는 두 가지 평가 축을 사용합니다.

| Content Preservation ↑ | Expressive Independence ↑ |
|---|---|
| Entity F1 | chrF++ ↓ |
| Numeric/Date F1 | Morpheme ROUGE-L ↓ |
| BERTScore F1 | Extractive Coverage ↓ |
| | Novel-4 ↑ |

추가적으로 두 평가 축의 조화평균인 **Balance Score**를 계산합니다.

## Reproduction

### Environment Setup

```bash
git clone https://github.com/on22y/CAPER.git
cd CAPER

conda create -n hclt python=3.10 -y
conda activate hclt
pip install -r requirements.txt
```

### Model Preparation

Pretrained model weights 및 DPO adapter weights는 저장소에 포함되어 있지 않으므로 별도로 준비해야 합니다.

필요한 경우 다음 환경변수로 checkpoint 경로를 지정할 수 있습니다.

```bash
export HCLT_BASE_MODEL=/path/to/HyperCLOVAX-SEED-Think-14B
export HCLT_DPO_ADAPTER=/path/to/dpo501_final_adapter
export OPENAI_API_KEY=YOUR_API_KEY
```

### Reproduction Guides

- [Multi-Model Comparison](docs/multimodel_comparison.md)
- [Experiment Provenance](docs/experiment_provenance.md)
- [Final Artifact Manifest](FINAL_FILE_MANIFEST.json)

`FINAL_FILE_MANIFEST.json`에는 주요 실험 산출물의 provenance 및 SHA256 checksum이 기록되어 있습니다.

본 저장소에는 pretrained model weights, LoRA/checkpoint weights, Hugging Face cache 및 API credentials를 포함하지 않습니다.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{lee2026caper,
  title     = {CAPER: Copyright-Aware Preference Learning for Expressively Independent Document Rewriting},
  author    = {Lee, Gayeon and Jang, Uk and Kim, Jihie},
  booktitle = {Proceedings of the 38th Annual Conference on Human and Cognitive Language Technology (HCLT 2026)},
  year      = {2026}
}
```
