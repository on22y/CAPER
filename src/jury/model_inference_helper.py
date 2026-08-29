#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    Mistral3ForConditionalGeneration,
)

from document_validation import parse_json_object, validate_document_response


ROOT = Path(__file__).resolve().parents[1]


class MistralCommonAdapter:
    def __init__(self, model_path: str, model_name: str) -> None:
        self.tokenizer = MistralTokenizer.from_file(str(Path(model_path) / "tekken.json"))
        self.model_name = model_name
        self.eos_token_id = 2

    def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> dict[str, torch.Tensor]:
        request = ChatCompletionRequest(
            messages=[UserMessage(content=messages[0]["content"])], model=self.model_name
        )
        tokens = self.tokenizer.encode_chat_completion(request).tokens
        input_ids = torch.tensor([tokens], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return self.tokenizer.decode(token_ids.tolist())


def install_exaone_mask_compatibility_shim() -> None:
    from transformers import masking_utils

    original = masking_utils.create_causal_mask

    def compatible_create_causal_mask(
        config: Any,
        input_embeds: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        position_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> Any:
        return original(
            config=config,
            inputs_embeds=inputs_embeds if inputs_embeds is not None else input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    masking_utils.create_causal_mask = compatible_create_causal_mask


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(row, ensure_ascii=False) + "\n")
        target.flush()
        os.fsync(target.fileno())


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        row["evaluation_id"]
        for row in read_jsonl(path)
        if row.get("validation_result", {}).get("status") == "valid"
        and not row.get("generation_reached_limit")
        and not row.get("error")
    }


def resolve_model_path(spec: dict[str, Any]) -> tuple[str, str]:
    model_path = spec.get("local_path")
    if not model_path:
        model_path = snapshot_download(spec["hf_id"], local_files_only=True)
    return str(model_path), Path(model_path).name


def load_model(spec: dict[str, Any]) -> tuple[Any, Any, str, str]:
    model_path, revision = resolve_model_path(spec)
    common = {
        "pretrained_model_name_or_path": model_path,
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "local_files_only": True,
        "trust_remote_code": bool(spec.get("trust_remote_code", False)),
        "low_cpu_mem_usage": True,
    }
    family = spec["family"]
    if family == "causal_lm":
        adapter = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
        )
        if spec["hf_id"].startswith("LGAI-EXAONE/"):
            install_exaone_mask_compatibility_shim()
        model = AutoModelForCausalLM.from_pretrained(**common)
    elif family == "gemma3":
        adapter = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = Gemma3ForConditionalGeneration.from_pretrained(**common)
    elif family == "ministral3":
        adapter = MistralCommonAdapter(model_path, spec["hf_id"])
        model = Mistral3ForConditionalGeneration.from_pretrained(**common)
    else:
        raise ValueError(f"unsupported family: {family}")
    model.eval()
    return model, adapter, model_path, revision


def input_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def format_inputs(adapter: Any, prompt: str, spec: dict[str, Any]) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    if spec.get("enable_thinking") is False:
        kwargs["enable_thinking"] = False
    try:
        encoded = adapter.apply_chat_template(messages, **kwargs)
    except (TypeError, ValueError):
        encoded = adapter.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}], **kwargs
        )
    if isinstance(encoded, torch.Tensor):
        encoded = {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
    return encoded


def decode_generated(adapter: Any, generated: torch.Tensor) -> str:
    if hasattr(adapter, "decode"):
        return adapter.decode(generated, skip_special_tokens=True)
    return adapter.tokenizer.decode(generated, skip_special_tokens=True)


def base_record(
    config: dict[str, Any], spec: dict[str, Any], model_name: str, row: dict[str, Any],
    model_path: str | None, revision: str | None, attempt: int,
) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "evaluation_id": row["evaluation_id"],
        "model_name": model_name,
        "model_hf_id": spec["hf_id"],
        "model_path": model_path,
        "model_revision": revision,
        "prompt_version": config["prompt_version"],
        "schema_version": config["schema_version"],
        "decoding": {"do_sample": False, "max_new_tokens": config["max_new_tokens"]},
        "attempt": attempt,
        "retry_count": attempt - 1,
        "retry_reason": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Document-level copyright jury evaluator")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--evaluation-id", action="append")
    parser.add_argument("--max-retries", type=int)
    args = parser.parse_args()

    config = json.loads((ROOT / "config_v5_document.json").read_text(encoding="utf-8"))
    if args.model not in config["models"]:
        raise ValueError(f"unknown model: {args.model}")
    if args.max_retries is not None:
        config["max_retries"] = max(0, args.max_retries)
    spec = config["models"][args.model]
    prompt_template = (ROOT / "prompts" / "jury_v5_document.txt").read_text(encoding="utf-8")
    output_path = args.output_dir / f"responses_v5_{args.model}.jsonl"
    rows = read_jsonl(args.input)
    if args.evaluation_id:
        wanted = set(args.evaluation_id)
        rows = [row for row in rows if row["evaluation_id"] in wanted]
    if args.limit is not None:
        rows = rows[:args.limit]
    done = completed_ids(output_path)
    rows = [row for row in rows if row["evaluation_id"] not in done]
    for row in read_jsonl(args.input):
        if not {"evaluation_id", "source_text", "target_text"} <= set(row):
            raise ValueError("input rows require evaluation_id, source_text, target_text")

    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    try:
        model, adapter, model_path, revision = load_model(spec)
    except Exception as error:  # noqa: BLE001
        for row in rows:
            record = base_record(config, spec, args.model, row, None, None, 1)
            record.update({
                "parse_status": "model_load_error",
                "validation_result": {"status": "invalid", "errors": [], "warnings": [], "metrics": {}},
                "error": f"{type(error).__name__}: {error}",
            })
            append_jsonl(output_path, record)
        raise

    device = input_device(model)
    for row in rows:
        prompt = prompt_template.replace("{source_text}", row["source_text"]).replace(
            "{target_text}", row["target_text"]
        )
        retry_reason: str | None = None
        retry_feedback: list[str] = []
        for attempt in range(1, int(config["max_retries"]) + 2):
            started = time.time()
            record = base_record(config, spec, args.model, row, model_path, revision, attempt)
            record["retry_reason"] = retry_reason
            retryable = False
            try:
                attempt_prompt = prompt
                if retry_feedback:
                    attempt_prompt += (
                        "\n\n[이전 출력 교정 요청]\n"
                        "이전 JSON이 아래 검증 오류로 거부되었습니다. 판단을 새로 만들지 말고 "
                        "같은 문서 비교를 유지하면서 오류를 모두 고친 완전한 JSON 객체를 다시 출력하세요.\n- "
                        + "\n- ".join(retry_feedback[:8])
                    )
                encoded = format_inputs(adapter, attempt_prompt, spec)
                encoded = {
                    key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)
                }
                input_tokens = int(encoded["input_ids"].shape[-1])
                with torch.inference_mode():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=int(config["max_new_tokens"]),
                        do_sample=False,
                        pad_token_id=getattr(adapter, "eos_token_id", None)
                        or getattr(getattr(adapter, "tokenizer", None), "eos_token_id", None),
                    )
                generated = output[0, input_tokens:]
                raw = decode_generated(adapter, generated)
                output_tokens = int(generated.shape[-1])
                reached_limit = output_tokens >= int(config["max_new_tokens"])
                parsed, parse_status, parse_error = parse_json_object(raw)
                if parsed is None:
                    validation = {
                        "status": "invalid",
                        "errors": [parse_error or parse_status],
                        "warnings": ["generation_reached_max_new_tokens:possible_truncation"] if reached_limit else [],
                        "metrics": {},
                    }
                else:
                    validation = validate_document_response(
                        parsed, row["source_text"], row["target_text"],
                        generation_reached_limit=reached_limit,
                    )
                record.update({
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_seconds": round(time.time() - started, 3),
                    "raw_response": raw,
                    "parsed_response": parsed,
                    "parse_status": parse_status,
                    "generation_reached_limit": reached_limit,
                    "validation_result": validation,
                })
                retryable = (
                    reached_limit
                    or parse_status in {"empty_response", "json_parse_error"}
                    or validation["status"] == "invalid"
                )
                retry_reason = (
                    "output_token_limit"
                    if reached_limit
                    else parse_status
                    if parse_status in {"empty_response", "json_parse_error"}
                    else "validation_failed"
                    if validation["status"] == "invalid"
                    else None
                )
                retry_feedback = list(validation.get("errors", []))
            except Exception as error:  # noqa: BLE001
                record.update({
                    "latency_seconds": round(time.time() - started, 3),
                    "raw_response": None,
                    "parsed_response": None,
                    "parse_status": "runtime_error",
                    "generation_reached_limit": False,
                    "validation_result": {"status": "invalid", "errors": [], "warnings": [], "metrics": {}},
                    "error": f"{type(error).__name__}: {error}",
                })
            append_jsonl(output_path, record)
            status = record["validation_result"]["status"]
            print(f"[{status}] {args.model} {row['evaluation_id']} attempt={attempt}", flush=True)
            if not retryable or attempt > int(config["max_retries"]):
                break


if __name__ == "__main__":
    main()
