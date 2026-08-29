# DPO 501 Training Summary

- status: `COMPLETED`
- input: `/home/user20250805/gayeon/HCLT2026/work/hcx_dpo900_mode_jury_v1/outputs/preferences/dpo_preferences_all_survivors.jsonl`
- dataset_rows_loaded: `501`
- base_model: `/home/user20250805/gayeon/HCLT2026/models/HyperCLOVAX-SEED-Think-14B`
- fresh_lora_initialized: `True`
- old_dpo_adapter_loaded: `False`
- parameter_counts: `{'total': 14809126912, 'trainable': 61014016, 'trainable_ratio': 0.004120027896483192, 'all_trainable_param_names_are_lora': True, 'trainable_param_name_examples': ['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight', 'base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight', 'base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight', 'base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight', 'base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight', 'base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight', 'base_model.model.model.layers.0.self_attn.o_proj.lora_A.default.weight', 'base_model.model.model.layers.0.self_attn.o_proj.lora_B.default.weight', 'base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight', 'base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight', 'base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight', 'base_model.model.model.layers.0.mlp.up_proj.lora_B.default.weight', 'base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight', 'base_model.model.model.layers.0.mlp.down_proj.lora_B.default.weight', 'base_model.model.model.layers.1.self_attn.q_proj.lora_A.default.weight', 'base_model.model.model.layers.1.self_attn.q_proj.lora_B.default.weight', 'base_model.model.model.layers.1.self_attn.k_proj.lora_A.default.weight', 'base_model.model.model.layers.1.self_attn.k_proj.lora_B.default.weight', 'base_model.model.model.layers.1.self_attn.v_proj.lora_A.default.weight', 'base_model.model.model.layers.1.self_attn.v_proj.lora_B.default.weight']}`
- elapsed_sec: `9308.053`
- metrics: `{'train_runtime': 9297.728, 'train_samples_per_second': 0.161, 'train_steps_per_second': 0.04, 'total_flos': 5.623048379563131e+17, 'train_loss': 0.306577103972435, 'epoch': 3.0}`
- final_adapter: `/home/user20250805/gayeon/HCLT2026/work/hcx_dpo501_train_v3/checkpoints/final_adapter`
