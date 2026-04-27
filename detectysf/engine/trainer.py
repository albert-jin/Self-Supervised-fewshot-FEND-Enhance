"""Training and evaluation loop for DetectYSF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from detectysf.data.collators import prompt_collate_fn
from detectysf.data.fewshot_dataset import PromptNewsDataset, load_fewshot_split
from detectysf.data.graph_builder import GraphArtifacts, load_graph_artifacts
from detectysf.models import DetectYSF
from detectysf.utils.metrics import binary_classification_metrics
from detectysf.utils.seed import set_seed


@dataclass
class IterationResult:
    iteration: int
    prompt_metrics: Dict[str, float]
    aligned_metrics: Dict[str, float]


def _to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _choose_device(requested: str) -> torch.device:
    if requested.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sample_opposite_label_texts(
    labels: torch.Tensor,
    label_to_texts: Dict[int, List[str]],
    fallback_texts: List[str],
) -> List[str]:
    sampled: List[str] = []
    for label in labels.detach().cpu().tolist():
        opposite = 1 - int(label)
        pool = label_to_texts.get(opposite, [])
        if pool:
            idx = np.random.randint(0, len(pool))
            sampled.append(pool[idx])
        else:
            idx = np.random.randint(0, len(fallback_texts))
            sampled.append(fallback_texts[idx])
    return sampled


def _prepare_label_pool(texts: List[str], labels: List[int]) -> Dict[int, List[str]]:
    out = {0: [], 1: []}
    for t, y in zip(texts, labels):
        out[int(y)].append(t)
    return out


def _build_dataloaders(cfg, tokenizer, split):
    train_ds = PromptNewsDataset(
        texts=split.train_texts,
        labels=split.train_labels,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        template=cfg.prompt_template,
    )
    test_ds = PromptNewsDataset(
        texts=split.test_texts,
        labels=split.test_labels,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        template=cfg.prompt_template,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=prompt_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=prompt_collate_fn,
    )
    return train_loader, test_loader


def _run_graph_alignment(
    graph: Optional[GraphArtifacts],
    test_probs: torch.Tensor,
    n_shots: int,
    percentile: float,
) -> Optional[torch.Tensor]:
    if graph is None:
        return None

    probs = test_probs.clone()
    diff = torch.abs(probs[:, 0] - probs[:, 1]).detach().cpu().numpy()
    threshold = np.percentile(diff, percentile)

    mask_real = (probs[:, 0] - probs[:, 1]) >= threshold
    mask_fake = (probs[:, 1] - probs[:, 0]) >= threshold
    probs[mask_real, 0] = 1.0
    probs[mask_real, 1] = 0.0
    probs[mask_fake, 0] = 0.0
    probs[mask_fake, 1] = 1.0

    all_probs = torch.cat([graph.train_confidence, probs], dim=0)
    propagated = torch.matmul(graph.adjacency, all_probs)
    propagated = torch.matmul(graph.adjacency, propagated)
    return propagated[n_shots:]


def _evaluate(
    model: DetectYSF,
    test_loader: DataLoader,
    graph: Optional[GraphArtifacts],
    cfg,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    probs_all: List[torch.Tensor] = []
    y_pred_all: List[torch.Tensor] = []
    y_true_all: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = _to_device(batch, device)
            prompt_out = model.forward_prompt(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                mask_pos=batch["mask_pos"],
                labels=None,
            )
            class_logits = prompt_out["class_logits"]
            probs = F.softmax(class_logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)

            probs_all.append(probs)
            y_pred_all.append(pred)
            y_true_all.append(batch["labels"])

    probs_tensor = torch.cat(probs_all, dim=0)
    y_pred = torch.cat(y_pred_all, dim=0).detach().cpu().numpy()
    y_true = torch.cat(y_true_all, dim=0).detach().cpu().numpy()

    prompt_metrics = binary_classification_metrics(y_true, y_pred)
    aligned_metrics = prompt_metrics
    if cfg.use_graph_alignment and graph is not None:
        aligned_probs = _run_graph_alignment(
            graph=graph,
            test_probs=probs_tensor,
            n_shots=cfg.n_shots,
            percentile=cfg.pseudo_label_percentile,
        )
        if aligned_probs is not None:
            aligned_pred = torch.argmax(aligned_probs, dim=-1).detach().cpu().numpy()
            aligned_metrics = binary_classification_metrics(y_true, aligned_pred)

    return {
        "prompt": prompt_metrics,
        "aligned": aligned_metrics,
    }


def _train_one_iteration(cfg, iteration: int) -> IterationResult:
    set_seed(cfg.seed + iteration)
    device = _choose_device(cfg.device)

    split = load_fewshot_split(cfg.data_dir, cfg.dataset_name, cfg.n_shots)
    model = DetectYSF(
        model_name_or_path=cfg.model_name_or_path,
        label_words=cfg.label_words,
        contrastive_temperature=cfg.contrastive_temperature,
        contrastive_mode=cfg.contrastive_mode,
        noise_dim=cfg.noise_dim,
        mlp_hidden=cfg.mlp_hidden,
    ).to(device)

    train_loader, test_loader = _build_dataloaders(cfg, model.tokenizer, split)
    train_label_pool = _prepare_label_pool(split.train_texts, split.train_labels)

    graph = None
    if cfg.use_graph_alignment:
        graph = load_graph_artifacts(
            data_dir=cfg.data_dir,
            dataset_name=cfg.dataset_name,
            n_shots=cfg.n_shots,
            user_threshold=cfg.user_threshold,
            device=device,
        )

    main_params = (
        list(model.backbone.parameters())
        + list(model.noise_generator.parameters())
        + list(model.neg_text_generator.parameters())
    )
    main_optim = AdamW(main_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    disc_optim = AdamW(model.discriminator.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = max(1, len(train_loader) * cfg.n_epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        main_optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for _ in range(cfg.n_epochs):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch, device)

            prompt_out = model.forward_prompt(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                mask_pos=batch["mask_pos"],
                labels=batch["labels"],
            )
            class_loss = prompt_out["mlm_loss"]
            sentence_embeddings = prompt_out["sentence_embedding"]

            if cfg.contrastive_mode == "unsupervised":
                pos_out = model.forward_prompt(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    mask_pos=batch["mask_pos"],
                    labels=None,
                )
                contrastive_loss = model.compute_contrastive_loss(
                    anchor_embeddings=sentence_embeddings,
                    labels=batch["labels"],
                    mode="unsupervised",
                    positive_embeddings=pos_out["sentence_embedding"],
                )
            else:
                contrastive_loss = model.compute_contrastive_loss(
                    anchor_embeddings=sentence_embeddings,
                    labels=batch["labels"],
                    mode="supervised",
                    positive_embeddings=None,
                )

            neg_texts = _sample_opposite_label_texts(
                labels=batch["labels"],
                label_to_texts=train_label_pool,
                fallback_texts=split.train_texts,
            )
            neg_enc = model.tokenizer(
                neg_texts,
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            )
            neg_enc = {k: v.to(device) for k, v in neg_enc.items()}

            adv_losses = model.compute_adversarial_losses(
                real_embeddings=sentence_embeddings,
                neg_input_ids=neg_enc["input_ids"],
                neg_attention_mask=neg_enc["attention_mask"],
                neg_token_type_ids=neg_enc.get("token_type_ids"),
            )

            disc_optim.zero_grad()
            adv_losses["disc_loss"].backward()
            disc_optim.step()

            for p in model.discriminator.parameters():
                p.requires_grad = False
            adv_losses_main = model.compute_adversarial_losses(
                real_embeddings=sentence_embeddings,
                neg_input_ids=neg_enc["input_ids"],
                neg_attention_mask=neg_enc["attention_mask"],
                neg_token_type_ids=neg_enc.get("token_type_ids"),
            )

            adv_total = (
                cfg.adversarial_delta * adv_losses_main["gen_adv_loss"]
                + (1.0 - cfg.adversarial_delta)
                * cfg.feature_matching_weight
                * adv_losses_main["feature_matching_loss"]
            )
            total_loss = class_loss + cfg.contrastive_weight * contrastive_loss + cfg.adversarial_weight * adv_total

            main_optim.zero_grad()
            total_loss.backward()
            main_optim.step()
            scheduler.step()
            for p in model.discriminator.parameters():
                p.requires_grad = True

    metric_sets = _evaluate(model=model, test_loader=test_loader, graph=graph, cfg=cfg, device=device)
    return IterationResult(
        iteration=iteration,
        prompt_metrics=metric_sets["prompt"],
        aligned_metrics=metric_sets["aligned"],
    )


def run_experiment(cfg) -> List[IterationResult]:
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    results: List[IterationResult] = []
    for i in range(cfg.iters):
        results.append(_train_one_iteration(cfg, i))
    return results
