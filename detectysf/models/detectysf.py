"""Unified DetectYSF model wrapper."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .adversarial import BinConDiscriminator, LMNegGenerator, NoiseMLPGenerator, compute_adversarial_losses
from .contrastive import SentenceContrastiveLoss
from .prompt_backbone import PromptMLMBackbone


class DetectYSF(nn.Module):
    """DetectYSF: Prompt learning + contrastive + adversarial modules."""

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        label_words=None,
        contrastive_temperature: float = 0.05,
        contrastive_mode: str = "unsupervised",
        noise_dim: int = 100,
        mlp_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = PromptMLMBackbone(
            model_name_or_path=model_name_or_path,
            label_words=label_words,
        )
        hidden_size = self.backbone.hidden_size
        self.contrastive = SentenceContrastiveLoss(
            temperature=contrastive_temperature,
            mode=contrastive_mode,
        )
        self.noise_generator = NoiseMLPGenerator(
            noise_dim=noise_dim,
            hidden_dim=mlp_hidden,
            output_dim=hidden_size,
        )
        self.neg_text_generator = LMNegGenerator(model_name_or_path=model_name_or_path)
        self.discriminator = BinConDiscriminator(input_dim=hidden_size)

    @property
    def tokenizer(self):
        return self.backbone.tokenizer

    def forward_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        mask_pos: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            mask_positions=mask_pos,
            labels=labels,
        )

    def compute_contrastive_loss(
        self,
        anchor_embeddings: torch.Tensor,
        labels: torch.Tensor,
        mode: str = "unsupervised",
        positive_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mode == "unsupervised":
            if positive_embeddings is None:
                raise ValueError("Unsupervised contrastive loss requires positive embeddings.")
            out = self.contrastive(
                embeddings=anchor_embeddings,
                positive_embeddings=positive_embeddings,
                mode="unsupervised",
            )
        else:
            out = self.contrastive(
                embeddings=anchor_embeddings,
                labels=labels,
                mode="supervised",
            )
        return out["loss"]

    def compute_adversarial_losses(
        self,
        real_embeddings: torch.Tensor,
        neg_input_ids: Optional[torch.Tensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
        neg_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fake_embeddings = [self.noise_generator(real_embeddings.size(0), real_embeddings.device)]
        if neg_input_ids is not None and neg_attention_mask is not None:
            fake_lm = self.neg_text_generator(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                token_type_ids=neg_token_type_ids,
            )
            fake_embeddings.append(fake_lm)
        return compute_adversarial_losses(
            discriminator=self.discriminator,
            real_embeddings=real_embeddings,
            fake_embeddings=fake_embeddings,
        )

