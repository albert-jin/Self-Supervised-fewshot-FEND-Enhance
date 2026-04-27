"""Train DetectYSF under few-shot settings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectysf.engine import run_experiment
from detectysf.utils.config import Config, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/detectysf.yaml")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--n_shots", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--label_words", type=str, default=None, help="Comma-separated, e.g. real,fake")
    parser.add_argument("--use_graph_alignment", action="store_true")
    parser.add_argument("--disable_graph_alignment", action="store_true")
    return parser.parse_args()


def merge_cfg(base: Config, args) -> Config:
    if args.dataset_name is not None:
        base.dataset_name = args.dataset_name
    if args.n_shots is not None:
        base.n_shots = args.n_shots
    if args.iters is not None:
        base.iters = args.iters
    if args.n_epochs is not None:
        base.n_epochs = args.n_epochs
    if args.batch_size is not None:
        base.batch_size = args.batch_size
    if args.device is not None:
        base.device = args.device
    if args.label_words is not None:
        base.label_words = [w.strip() for w in args.label_words.split(",") if w.strip()]
    if args.use_graph_alignment:
        base.use_graph_alignment = True
    if args.disable_graph_alignment:
        base.use_graph_alignment = False
    return base


def summarize(results):
    prompt_acc = [r.prompt_metrics["accuracy"] for r in results]
    prompt_prec = [r.prompt_metrics["precision_macro"] for r in results]
    prompt_rec = [r.prompt_metrics["recall_macro"] for r in results]
    prompt_f1 = [r.prompt_metrics["f1_macro"] for r in results]

    align_acc = [r.aligned_metrics["accuracy"] for r in results]
    align_prec = [r.aligned_metrics["precision_macro"] for r in results]
    align_rec = [r.aligned_metrics["recall_macro"] for r in results]
    align_f1 = [r.aligned_metrics["f1_macro"] for r in results]

    return {
        "prompt_accuracy_mean": float(np.mean(prompt_acc)),
        "prompt_precision_macro_mean": float(np.mean(prompt_prec)),
        "prompt_recall_macro_mean": float(np.mean(prompt_rec)),
        "prompt_f1_macro_mean": float(np.mean(prompt_f1)),
        "aligned_accuracy_mean": float(np.mean(align_acc)),
        "aligned_precision_macro_mean": float(np.mean(align_prec)),
        "aligned_recall_macro_mean": float(np.mean(align_rec)),
        "aligned_f1_macro_mean": float(np.mean(align_f1)),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cfg(cfg, args)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    results = run_experiment(cfg)
    summary = summarize(results)

    print("DetectYSF few-shot summary")
    print(summary)

    log_path = Path(cfg.log_dir) / (
        "log_{dataset}_fewshot_{shots}_samples_DetectYSF.iter{iters}.txt".format(
            dataset=cfg.dataset_name,
            shots=cfg.n_shots,
            iters=cfg.iters,
        )
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Prompt template: {t}\n".format(t=cfg.prompt_template))
        f.write("Label words: {w}\n".format(w=",".join(cfg.label_words)))
        for r in results:
            f.write(
                "iter={i} prompt={pm} aligned={am}\n".format(
                    i=r.iteration,
                    pm=r.prompt_metrics,
                    am=r.aligned_metrics,
                )
            )
        f.write("summary={s}\n".format(s=summary))


if __name__ == "__main__":
    main()
