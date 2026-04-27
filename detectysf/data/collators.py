"""Batch collation utilities."""

from __future__ import annotations

from typing import Dict, List

import torch


def prompt_collate_fn(features: List[Dict]) -> Dict:
    """Collate variable dict features into a batch tensor dict."""

    batch: Dict = {
        "input_ids": torch.stack([x["input_ids"] for x in features], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in features], dim=0),
        "labels": torch.stack([x["labels"] for x in features], dim=0),
        "mask_pos": torch.stack([x["mask_pos"] for x in features], dim=0),
        "text": [x["text"] for x in features],
        "prompt_text": [x["prompt_text"] for x in features],
    }
    if "token_type_ids" in features[0]:
        batch["token_type_ids"] = torch.stack([x["token_type_ids"] for x in features], dim=0)
    return batch

