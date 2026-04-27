"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Config:
    data_dir: str = "data"
    dataset_name: str = "politifact"
    n_shots: int = 16
    model_name_or_path: str = "bert-base-uncased"
    label_words: List[str] = field(default_factory=lambda: ["real", "fake"])
    prompt_template: str = "It is {mask} that {text}."
    max_length: int = 512
    batch_size: int = 16
    n_epochs: int = 3
    iters: int = 20
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    user_threshold: int = 5
    use_graph_alignment: bool = True
    pseudo_label_percentile: float = 95.0
    contrastive_mode: str = "unsupervised"
    contrastive_temperature: float = 0.05
    contrastive_weight: float = 0.2
    adversarial_weight: float = 0.2
    adversarial_delta: float = 0.5
    feature_matching_weight: float = 1.0
    noise_dim: int = 100
    mlp_hidden: int = 512
    seed: int = 0
    num_workers: int = 0
    device: str = "cuda"
    log_dir: str = "logs"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        base = cls()
        for key, value in d.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base


def load_config(path: str) -> Config:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError("Config not found: {p}".format(p=cfg_path))

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required to load yaml config.") from exc

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f) or {}
    if not isinstance(cfg_dict, dict):
        raise ValueError("Config yaml must be a dictionary at top-level.")
    return Config.from_dict(cfg_dict)

