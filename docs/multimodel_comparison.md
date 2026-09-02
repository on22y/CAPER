# Multi-model DPO Comparison Runbook

This workflow starts from the frozen 498 preference pairs. It does not rebuild the 900-source
preference pipeline or rerun the local LLM jury.

## 1. Clone and environment

```bash
git clone https://github.com/on22y/HCLT_2026.git
cd HCLT_2026
conda create -n hclt python=3.10 -y
conda activate hclt
```

Install a stable PyTorch CUDA wheel supported by the new server driver before installing the
remaining requirements. The successful reference environment used PyTorch `2.11.0+cu128`;
the `CUDA 13.0` compatibility value printed by `nvidia-smi` does not require a CUDA 13.0 wheel.
Confirm the current command with the official PyTorch selector. An example matching the reference
build is:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 2. Hugging Face authentication and checkpoints

Gemma is gated. Accept its license with the Hugging Face account used on the server, then log in.

```bash
hf auth login
mkdir -p models/benchmarks models/evaluation
hf download Qwen/Qwen3-14B --local-dir models/benchmarks/Qwen3-14B
hf download google/gemma-3-12b-it --local-dir models/benchmarks/gemma-3-12b-it
```

The final automatic metrics require these exact backbones. The runner uses the Hugging Face IDs
by default and expects them in the local cache. Explicit directories can be supplied with the
environment variables shown below.

```bash
hf download chunwoolee0/klue_ner_roberta_model --local-dir models/evaluation/klue_ner_roberta_model
hf download joeddav/xlm-roberta-large-xnli --local-dir models/evaluation/xlm-roberta-large-xnli
hf download xlm-roberta-large --local-dir models/evaluation/xlm-roberta-large

export HCLT_NER_MODEL="$PWD/models/evaluation/klue_ner_roberta_model"
export HCLT_NLI_MODEL="$PWD/models/evaluation/xlm-roberta-large-xnli"
export HCLT_BERTSCORE_MODEL="$PWD/models/evaluation/xlm-roberta-large"
```

## 3. Frozen data

| Artifact | Rows | SHA256 |
|---|---:|---|
| `data/processed/dpo_train_498.jsonl` | 498 | `5e5e223946143342c6f268ae73e64d41d2aceb1354fa92156b6d99e9837ad5d1` |
| `data/processed/heldout_51.jsonl` | 51 | `8c66122c61c756a732d24576440dff0b8ec546338d4016769d563bb8623058f1` |

```bash
cd data/processed
sha256sum -c SHA256SUMS
cd ../..
```

The comparison file excludes three prompts that exceeded 3072 tokens under the HyperCLOVAX
preflight tokenizer. The successful HyperCLOVAX v3 run itself passed all 501 rows to TRL, which
truncated those rows; therefore `498` describes the frozen cross-model comparison set, not the
literal row count consumed by that historical run.

## 4. Shared settings

`configs/training/dpo_common.json` records the successful HyperCLOVAX settings: three epochs,
beta `0.1`, learning rate `5e-6`, max length `3072`, LoRA `r=16`, alpha `32`, dropout `0.05`, and
effective batch size `4`. The original two RTX 4090 run used model parallelism in one process, so
its effective batch was `1 x gradient accumulation 4`, not data-parallel batch multiplication.

On one RTX PRO 5000 72GB, keep `per_device_train_batch_size=1` and
`gradient_accumulation_steps=4`. Change both only if their product remains four.

Architecture-specific differences are limited to the official tokenizer chat template and LoRA
module discovery. Qwen disables thinking in the chat template. Gemma templates that reject a
system role receive the same system text folded into the user message. Gemma LoRA targets are
restricted to `language_model.layers.*`; vision-tower projections are excluded.

## 5. Qwen3-14B DPO

Run a two-step smoke test first, then use a separate output directory for full training.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python src/training/run_dpo.py \
  --model-name-or-path models/benchmarks/Qwen3-14B \
  --model-family qwen3 \
  --train-file data/processed/dpo_train_498.jsonl \
  --output-dir outputs/qwen3_14b/smoke \
  --smoke-steps 2

python src/training/run_dpo.py \
  --model-name-or-path models/benchmarks/Qwen3-14B \
  --model-family qwen3 \
  --train-file data/processed/dpo_train_498.jsonl \
  --output-dir outputs/qwen3_14b/dpo
```

## 6. Gemma-3-12B-it DPO

```bash
python src/training/run_dpo.py \
  --model-name-or-path models/benchmarks/gemma-3-12b-it \
  --model-family gemma3 \
  --train-file data/processed/dpo_train_498.jsonl \
  --output-dir outputs/gemma3_12b/smoke \
  --smoke-steps 2

python src/training/run_dpo.py \
  --model-name-or-path models/benchmarks/gemma-3-12b-it \
  --model-family gemma3 \
  --train-file data/processed/dpo_train_498.jsonl \
  --output-dir outputs/gemma3_12b/dpo
```

## 7. Matched Base/DPO held-out generation

Base and adapter generation use the same 51 samples, prompt, decoding configuration, sample seed,
and retry policy. Each model uses its official chat template.

```bash
python src/generation/run_base_vs_dpo_final51.py \
  --base-model models/benchmarks/Qwen3-14B \
  --model-family qwen3 \
  --adapter outputs/qwen3_14b/dpo/final_adapter \
  --input data/processed/heldout_51.jsonl \
  --output outputs/qwen3_14b/final51

python src/generation/run_base_vs_dpo_final51.py \
  --base-model models/benchmarks/gemma-3-12b-it \
  --model-family gemma3 \
  --adapter outputs/gemma3_12b/dpo/final_adapter \
  --input data/processed/heldout_51.jsonl \
  --output outputs/gemma3_12b/final51
```

## 8. Automatic metrics

```bash
python src/evaluation/run_comparison_metrics.py \
  --input outputs/qwen3_14b/final51/base_vs_dpo_rewrites_51.jsonl \
  --output-dir outputs/qwen3_14b/metrics

python src/evaluation/run_comparison_metrics.py \
  --input outputs/gemma3_12b/final51/base_vs_dpo_rewrites_51.jsonl \
  --output-dir outputs/gemma3_12b/metrics
```

The metric suite is unchanged from the final HyperCLOVAX evaluation: Entity F1, Numeric/Date F1,
BiNLI Support/Coverage/F1, Korean BERTScore P/R/F1, chrF++, Morpheme ROUGE-L, Extractive
Coverage/Density, and Novel-2/Novel-4. QA-based factuality remains excluded because the original
final run recorded it as blocked.

## 9. Output layout

```text
outputs/
  qwen3_14b/
    smoke/
    dpo/
    final51/
    metrics/
  gemma3_12b/
    smoke/
    dpo/
    final51/
    metrics/
```

The historical 900-source preference-construction code under `src/preference/` is provenance code.
It is not part of this comparison workflow and is not required on the new server.
