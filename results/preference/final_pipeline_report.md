# DPO900 Mode Jury Pipeline Report

- train sources: 900
- heldout sources frozen, unused after split: 51
- complete C/T pairs: 888
- expected jury judgments: 5328
- valid jury keys: {'qwen3': 1776, 'granite': 1776, 'ministral3': 1776}
- DPO survivors: 501
- survivor rate: 0.5642
- chosen identity counts: {'CONSERVATIVE': 45, 'TRANSFORMATIVE': 456}
- domain counts: {'news': 140, 'petition': 172, 'sns': 189}
- reject reasons: {'insufficient_stable_votes': 307, 'stable_conflict': 80}
- GPT/OpenAI calls: 0
