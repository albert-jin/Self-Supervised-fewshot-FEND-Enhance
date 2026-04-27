"""Prompt-based MLM backbone for DetectYSF."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer


class PromptMLMBackbone(nn.Module):
    """MLM prompt backbone using [MASK] token logits for binary classification."""

    SUPPORTED_LABEL_SETS = (("real", "fake"), ("news", "rumor"))

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        label_words: Optional[Sequence[str]] = None,
        pooler: str = "cls",
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.mlm = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.pooler = pooler
        self.hidden_size = int(self.mlm.config.hidden_size)

        default_words = ["real", "fake"]
        self.label_words: List[str] = list(label_words or default_words)
        self.register_buffer(
            "label_token_ids",
            torch.tensor(self._resolve_label_token_ids(self.label_words), dtype=torch.long),
            persistent=True,
        )

    def set_label_words(self, label_words: Sequence[str]) -> None:
        """Update label words in-place."""
        self.label_words = list(label_words)
        token_ids = torch.tensor(self._resolve_label_token_ids(self.label_words), dtype=torch.long)
        self.label_token_ids = token_ids.to(self.label_token_ids.device)

    def _resolve_label_token_ids(self, label_words: Sequence[str]) -> List[int]:
        if len(label_words) != 2:
            raise ValueError(f"Binary classification needs exactly 2 label words, got {label_words}.")

        token_ids: List[int] = []
        for word in label_words:
            token_ids.append(self._resolve_single_token_id(word))
        return token_ids

    def _resolve_single_token_id(self, word: str) -> int:
        candidates = [word]
        if not word.startswith(" "):
            candidates.append(f" {word}")

        for candidate in candidates:
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1 and ids[0] != self.tokenizer.unk_token_id:
                return int(ids[0])

        fallback = self.tokenizer.encode(word, add_special_tokens=False)
        if not fallback:
            raise ValueError(f"Failed to tokenize label word '{word}' into any token id.")
        return int(fallback[0])

    def _find_mask_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("Tokenizer has no mask token id, cannot run prompt MLM.")

        has_mask = (input_ids == mask_token_id).any(dim=1)
        if not bool(has_mask.all()):
            bad_rows = (~has_mask).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"Input rows without [MASK] token found at indices: {bad_rows}")

        # Pick the first [MASK] position in each sample.
        return (input_ids == mask_token_id).long().argmax(dim=1)

    def _pool_sentence_embedding(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.pooler == "mean":
            if attention_mask is None:
                return last_hidden_state.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (last_hidden_state * mask).sum(dim=1) / denom
        return last_hidden_state[:, 0]

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode sentence embeddings (CLS/mean pooled) without computing prompt logits."""
        kwargs: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.mlm(**kwargs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        return self._pool_sentence_embedding(last_hidden_state, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
            "output_hidden_states": True,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.mlm(**kwargs)
        last_hidden_state = outputs.hidden_states[-1]

        if mask_positions is None:
            mask_positions = self._find_mask_positions(input_ids)
        mask_positions = mask_positions.to(input_ids.device)

        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        mask_hidden_state = last_hidden_state[batch_indices, mask_positions]
        mask_vocab_logits = outputs.logits[batch_indices, mask_positions]

        label_token_ids = self.label_token_ids.to(mask_vocab_logits.device)
        class_logits = mask_vocab_logits.index_select(dim=1, index=label_token_ids)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(class_logits, labels.long())

        sentence_embedding = self._pool_sentence_embedding(last_hidden_state, attention_mask)

        return {
            "class_logits": class_logits,
            "mlm_loss": loss,
            "sentence_embedding": sentence_embedding,
            "mask_hidden_state": mask_hidden_state,
            "mask_vocab_logits": mask_vocab_logits,
            "label_token_ids": label_token_ids,
            "mask_positions": mask_positions,
        }
