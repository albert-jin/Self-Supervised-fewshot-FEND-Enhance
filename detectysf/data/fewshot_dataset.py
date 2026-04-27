"""Few-shot dataset loading and prompt-ready torch Dataset definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class FewShotSplit:
    """Container for one few-shot split."""

    train_texts: List[str]
    train_labels: List[int]
    test_texts: List[str]
    test_labels: List[int]


def _load_label_text_csv(path: Path) -> Dict[str, List]:
    frame = pd.read_csv(path, header=None, names=["label", "text"])
    labels = frame["label"].astype(int).tolist()
    texts = frame["text"].fillna("").astype(str).tolist()
    return {"labels": labels, "texts": texts}


def load_fewshot_split(data_dir: str, dataset_name: str, n_shots: int) -> FewShotSplit:
    """Load few-shot train/test split from prepared csv files."""

    root = Path(data_dir)
    train_path = root / "{name}_train_{k}.csv".format(name=dataset_name, k=n_shots)
    test_path = root / "{name}_test.csv".format(name=dataset_name)
    if not train_path.exists():
        raise FileNotFoundError("Cannot find train file: {p}".format(p=train_path))
    if not test_path.exists():
        raise FileNotFoundError("Cannot find test file: {p}".format(p=test_path))

    train = _load_label_text_csv(train_path)
    test = _load_label_text_csv(test_path)
    return FewShotSplit(
        train_texts=train["texts"],
        train_labels=train["labels"],
        test_texts=test["texts"],
        test_labels=test["labels"],
    )


class PromptNewsDataset(Dataset):
    """Torch dataset that converts news text into prompt-MLM inputs."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
        template: str = "It is {mask} that {text}.",
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts/labels length mismatch: {a} vs {b}".format(a=len(texts), b=len(labels)))
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template

    def __len__(self) -> int:
        return len(self.texts)

    def _build_prompt(self, text: str) -> str:
        return self.template.format(mask=self.tokenizer.mask_token, text=text)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        label = int(self.labels[idx])
        text = self.texts[idx]
        prompt = self._build_prompt(text)
        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(0)

        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            # Fallback: if tokenizer unexpectedly altered prompt, use position 0.
            mask_pos = torch.tensor(0, dtype=torch.long)
        else:
            mask_pos = mask_positions[0, 0].long()

        item = {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "labels": torch.tensor(label, dtype=torch.long),
            "mask_pos": mask_pos,
            "text": text,
            "prompt_text": prompt,
        }
        if token_type_ids is not None:
            item["token_type_ids"] = token_type_ids.long()
        return item

