"""Contrastive objectives for sentence representations."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceContrastiveLoss(nn.Module):
    """Contrastive loss with cosine similarity and in-batch negatives."""

    def __init__(self, temperature: float = 0.07, mode: str = "unsupervised") -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.temperature = float(temperature)
        self.mode = mode

    def _unsupervised_loss(
        self, embeddings: torch.Tensor, positive_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z1 = F.normalize(embeddings, dim=-1)
        z2 = F.normalize(positive_embeddings, dim=-1)

        sim = torch.matmul(z1, z2.transpose(0, 1)) / self.temperature
        targets = torch.arange(z1.size(0), device=z1.device)

        loss_12 = F.cross_entropy(sim, targets)
        loss_21 = F.cross_entropy(sim.transpose(0, 1), targets)
        loss = 0.5 * (loss_12 + loss_21)

        return {"loss": loss, "similarity": sim}

    def _supervised_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if embeddings.dim() == 3:
            batch_size, n_views, dim = embeddings.shape
            embeddings = embeddings.view(batch_size * n_views, dim)
            labels = labels.view(-1, 1).repeat(1, n_views).view(-1)

        z = F.normalize(embeddings, dim=-1)
        sim = torch.matmul(z, z.transpose(0, 1)) / self.temperature

        logits_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.transpose(0, 1)) & logits_mask
        positive_count = positive_mask.sum(dim=1)

        mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1) / (positive_count + 1e-12)
        valid = positive_count > 0
        if bool(valid.any()):
            loss = -mean_log_prob_pos[valid].mean()
        else:
            loss = sim.new_zeros(())

        return {"loss": loss, "similarity": sim, "positive_count": positive_count}

    def forward(
        self,
        embeddings: torch.Tensor,
        positive_embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        active_mode = mode or self.mode
        if active_mode == "unsupervised":
            if positive_embeddings is None:
                if embeddings.dim() == 3 and embeddings.size(1) >= 2:
                    positive_embeddings = embeddings[:, 1]
                    embeddings = embeddings[:, 0]
                else:
                    raise ValueError("Unsupervised mode needs positive_embeddings or 2-view embeddings.")
            return self._unsupervised_loss(embeddings, positive_embeddings)

        if active_mode == "supervised":
            if labels is None:
                raise ValueError("Supervised mode requires labels.")
            return self._supervised_loss(embeddings, labels.long())

        raise ValueError(f"Unsupported contrastive mode: {active_mode}")
