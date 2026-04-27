"""Build Prompt-and-Align style news proximity graph from raw social data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectysf.data.graph_builder import build_news_proximity_graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["politifact", "gossipcop", "fang"])
    parser.add_argument("--n_shots", type=int, required=True, choices=[16, 32, 64, 128])
    parser.add_argument("--user_threshold", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="data/adjs_from_scratch")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = build_news_proximity_graph(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        n_shots=args.n_shots,
        user_threshold=args.user_threshold,
        output_dir=args.output_dir,
    )
    print("saved:", out_path)


if __name__ == "__main__":
    main()
