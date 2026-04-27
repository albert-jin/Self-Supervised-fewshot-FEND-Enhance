"""News proximity graph loading/building utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


@dataclass
class GraphArtifacts:
    """Graph artifacts consumed by social alignment stage."""

    train_confidence: torch.Tensor
    adjacency: torch.Tensor
    adjacency_sparse: sp.spmatrix


def _normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
    rowsum = np.asarray(adj.sum(1)).flatten()
    d_row = np.power(rowsum, -0.5)
    d_row[np.isinf(d_row)] = 0.0
    d_row_mat = sp.diags(d_row)

    colsum = np.asarray(adj.sum(0)).flatten()
    d_col = np.power(colsum, -0.5)
    d_col[np.isinf(d_col)] = 0.0
    d_col_mat = sp.diags(d_col)

    return adj.dot(d_col_mat).transpose().dot(d_row_mat).transpose()


def _train_confidence_from_split(data_dir: Path, dataset_name: str, n_shots: int) -> np.ndarray:
    frame = pd.read_csv(
        data_dir / "{name}_train_{k}.csv".format(name=dataset_name, k=n_shots),
        header=None,
        names=["label", "text"],
    )
    labels = frame["label"].astype(int).tolist()
    train_conf = np.zeros((len(labels), 2), dtype=np.float32)
    for idx, label in enumerate(labels):
        if label == 0:
            train_conf[idx] = np.array([1.0, 0.0], dtype=np.float32)
        else:
            train_conf[idx] = np.array([0.0, 1.0], dtype=np.float32)
    return train_conf


def build_news_proximity_graph(
    data_dir: str,
    dataset_name: str,
    n_shots: int,
    user_threshold: int = 5,
    output_dir: str = "data/adjs_from_scratch",
) -> Path:
    """
    Rebuild Prompt-and-Align style news-news proximity graph.

    Saved payload is compatible with existing pkl format:
    [train_confidence(ndarray), adjacency(scipy sparse)].
    """

    root = Path(data_dir)
    news_path = root / "news_articles_raw" / "{name}_full_train{k}.csv".format(
        name=dataset_name, k=n_shots
    )
    social_path = root / "social_context_raw" / "{name}_socialcontext_train{k}.csv".format(
        name=dataset_name, k=n_shots
    )
    if not news_path.exists():
        raise FileNotFoundError("Cannot find raw news file: {p}".format(p=news_path))
    if not social_path.exists():
        raise FileNotFoundError("Cannot find raw social file: {p}".format(p=social_path))

    news_df = pd.read_csv(news_path, encoding="utf-8")
    social_df = pd.read_csv(social_path, encoding="utf-8")

    news_ids: List[str] = news_df["news_id"].astype(str).tolist()
    sid_list: List[str] = social_df["sid"].astype(str).tolist()
    user_list: List[str] = social_df["uid"].astype(str).tolist()
    sid_unique: List[str] = list(dict.fromkeys(sid_list))

    uid_for_known_news: List[str] = []
    for sid, uid in zip(sid_list, user_list):
        if sid in news_ids:
            uid_for_known_news.append(uid)

    active_counter = Counter({u: c for u, c in Counter(uid_for_known_news).items() if c >= user_threshold})
    active_users = list(active_counter.keys())

    user_to_idx: Dict[str, int] = {u: i for i, u in enumerate(active_users)}
    sid_to_idx: Dict[str, int] = {s: i for i, s in enumerate(sid_unique)}

    user_news_pairs: List[Tuple[str, str]] = []
    freq_news: List[str] = []
    for sid, uid in zip(sid_list, user_list):
        if uid in active_users:
            user_news_pairs.append((sid, uid))
            freq_news.append(sid)

    propagated_news = list(dict.fromkeys(freq_news))
    not_propagated = list(set(sid_unique) ^ set(propagated_news))

    # Ensure every news node can connect with at least one pseudo user.
    for i, sid in enumerate(not_propagated):
        pseudo_uid = "pseudo_{i}".format(i=i)
        user_to_idx[pseudo_uid] = len(active_users) + i
        user_news_pairs.append((sid, pseudo_uid))

    relation = np.asarray(
        [(user_to_idx[u], sid_to_idx[s], 1.0) for s, u in user_news_pairs], dtype=np.float32
    )
    user_news = sp.csc_matrix(
        (relation[:, 2], (relation[:, 0], relation[:, 1])),
        shape=(len(active_users) + len(not_propagated), len(sid_unique)),
        dtype=np.float32,
    )
    adj = user_news.transpose().dot(user_news)
    adj = _normalize_adj(adj)

    train_conf = _train_confidence_from_split(root, dataset_name, n_shots)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "{name}_nn_relations_{k}.pkl".format(name=dataset_name, k=n_shots)
    # Local import to avoid forcing pickle on module import.
    import pickle

    with open(out_path, "wb") as fp:
        pickle.dump([train_conf, adj], fp)
    return out_path


def load_graph_artifacts(
    data_dir: str,
    dataset_name: str,
    n_shots: int,
    user_threshold: int,
    device: torch.device,
) -> GraphArtifacts:
    """Load prepared adjacency and training confidence from pkl."""

    import pickle

    root = Path(data_dir)
    pkl_path = root / "adjs" / "user_t{t}".format(t=user_threshold) / "{name}_nn_relations_{k}.pkl".format(
        name=dataset_name, k=n_shots
    )
    if not pkl_path.exists():
        raise FileNotFoundError(
            "Cannot find adjacency file: {p}. Run scripts/build_graph.py first.".format(p=pkl_path)
        )

    with open(pkl_path, "rb") as fp:
        train_conf, adj = pickle.load(fp)

    train_conf_t = torch.tensor(np.asarray(train_conf), dtype=torch.float32, device=device)
    adjacency_dense = torch.tensor(np.asarray(adj.todense()), dtype=torch.float32, device=device)
    return GraphArtifacts(
        train_confidence=train_conf_t,
        adjacency=adjacency_dense,
        adjacency_sparse=adj,
    )
