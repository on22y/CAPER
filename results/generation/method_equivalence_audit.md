# Method Equivalence Audit

| Item | Old | New | Match? |
|---|---|---|---|
| Base model | HyperCLOVAX-SEED-Think-14B | HyperCLOVAX-SEED-Think-14B | YES |
| Tokenizer | local native tokenizer | local native tokenizer | YES |
| Chat template | native apply_chat_template, skip_reasoning=True | same | YES |
| Prompt SHA | ff5528...a798 | ff5528...a798 | YES |
| do_sample | true | true | YES |
| temperature | 0.7 | 0.7 | YES |
| top_p | 0.9 | 0.9 | YES |
| repetition_penalty | 1.05 | 1.05 | YES |
| max_new_tokens | 1536 | 1536 | YES |
| seed policy | SHA256(sample_id + heldout rewrite suffix) | same | YES |
| JSON parser | canonical strict parser | copied unchanged | YES |
| validation | canonical strict validation | copied unchanged | YES |
| retry policy | max 3, deterministic retry seed | same | YES |
| Held-out path | old held-out60 | final untouched held-out60 | INTENDED DIFFERENCE |
| DPO adapter | pilot186 adapter | expanded367 adapter | INTENDED DIFFERENCE |
| Output root | old evaluation root | new final evaluation root | INTENDED DIFFERENCE |

Result: PASS. No unintended generation-method difference found.
