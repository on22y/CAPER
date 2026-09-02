#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CANONICAL_RUNNER = REPO / "src/evaluation/run_automatic_metrics.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("hclt_canonical_automatic_metrics", CANONICAL_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import canonical runner: {CANONICAL_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the final HCLT metric suite to a Base-vs-DPO final51 file")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runner = load_runner()
    runner.INPUT = args.input.resolve()
    runner.ROOT = args.output_dir.resolve()
    runner.PER_SAMPLE = runner.ROOT / "per_sample_metrics_102.jsonl"
    runner.SUMMARY = runner.ROOT / "automatic_metric_summary.json"
    runner.REPORT = runner.ROOT / "automatic_metric_summary.md"
    runner.MANIFEST = runner.ROOT / "automatic_metric_manifest.json"
    runner.main()


if __name__ == "__main__":
    main()
