# DPO 501 Length Audit

- canonical input rows: `501`
- effective training rows: `498`
- prompt-length dropped: `3`
- completion partially truncated: `17`
- estimated optimization steps: `375`

## Prompt-Length Drops

[
  {
    "sample_id": "dpo900_train_petition_074",
    "domain": "petition",
    "prompt": 3496,
    "chosen_completion": 651,
    "rejected_completion": 431,
    "prompt_chosen": 4147,
    "prompt_rejected": 3927
  },
  {
    "sample_id": "dpo900_train_petition_083",
    "domain": "petition",
    "prompt": 3774,
    "chosen_completion": 1287,
    "rejected_completion": 1369,
    "prompt_chosen": 5061,
    "prompt_rejected": 5143
  },
  {
    "sample_id": "dpo900_train_petition_210",
    "domain": "petition",
    "prompt": 3598,
    "chosen_completion": 1107,
    "rejected_completion": 1113,
    "prompt_chosen": 4705,
    "prompt_rejected": 4711
  }
]
