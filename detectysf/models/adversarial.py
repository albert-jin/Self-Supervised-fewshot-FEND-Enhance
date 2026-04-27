"""Adversarial learning modules for DetectYSF."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class NoiseMLPGenerator(nn.Module):
    """Noise -> embedding generator (100 -> 512 -> hidden)."""

    def __init__(
        self,
        noise_dim: int = 100,
        hidden_dim: int = 512,
        output_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch_size, self.noise_dim, device=device)
        return self.net(z)


class LMNegGenerator(nn.Module):
    """Independent LM encoder used to generate fabricated embeddings from negative text."""

    def __init__(self, model_name_or_path: str = "bert-base-uncased", pooler: str = "cls") -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.pooler = pooler
        self.hidden_size = int(self.encoder.config.hidden_size)

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooler == "mean":
            if attention_mask is None:
                return last_hidden_state.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (last_hidden_state * mask).sum(dim=1) / denom
        return last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        return self._pool(outputs.last_hidden_state, attention_mask)


class BinConDiscriminator(nn.Module):
    """Binary real/fabricated discriminator."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def _stack_fake_embeddings(fake_embeddings: List[torch.Tensor]) -> torch.Tensor:
    if not fake_embeddings:
        raise ValueError("fake_embeddings cannot be empty.")
    return torch.cat(fake_embeddings, dim=0)


def compute_adversarial_losses(
    discriminator: BinConDiscriminator,
    real_embeddings: torch.Tensor,
    fake_embeddings: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute discriminator, generator fooling and feature matching losses.

    - disc_loss: train D to classify real(1)/fake(0)
    - gen_adv_loss: train G to fool D as real(1)
    - feature_matching_loss: L2 distance between fake and real embedding stats
    """

    device = real_embeddings.device
    all_fake = _stack_fake_embeddings(fake_embeddings)

    real_targets = torch.ones(real_embeddings.size(0), dtype=torch.long, device=device)
    fake_targets = torch.zeros(all_fake.size(0), dtype=torch.long, device=device)

    # Discriminator optimization objective (detach generated and real signals).
    real_logits_d = discriminator(real_embeddings.detach())
    fake_logits_d = discriminator(all_fake.detach())
    disc_loss = F.cross_entropy(real_logits_d, real_targets) + F.cross_entropy(fake_logits_d, fake_targets)

    # Generator objective: fool discriminator into predicting "real".
    fake_logits_g = discriminator(all_fake)
    gen_adv_loss = F.cross_entropy(fake_logits_g, torch.ones(all_fake.size(0), dtype=torch.long, device=device))

    # Feature matching towards real sample manifold.
    real_mean = real_embeddings.mean(dim=0, keepdim=True)
    feature_matching_loss = all_fake.new_zeros(())
    for fake in fake_embeddings:
        feature_matching_loss = feature_matching_loss + F.mse_loss(fake, real_mean.expand_as(fake))

    return {
        "disc_loss": disc_loss,
        "gen_adv_loss": gen_adv_loss,
        "feature_matching_loss": feature_matching_loss,
    }

