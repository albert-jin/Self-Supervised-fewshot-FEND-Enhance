# Self-Supervised-fewshot-FEND-Enhance

Code reproduction for:
"A veracity dissemination consistency-based few-shot fake news detection framework by synergizing adversarial and contrastive self-supervised learning"

## Project Layout

- `detectysf/data`
  - few-shot split loading (`data/{dataset}_train_{k}.csv`, `data/{dataset}_test.csv`)
  - prompt dataset building
  - news proximity graph loading/building
- `detectysf/models`
  - prompt MLM backbone
  - sentence-level contrastive loss
  - adversarial generators/discriminator
  - unified DetectYSF model wrapper
- `detectysf/engine`
  - training loop (few-shot iterative runs)
  - evaluation and social graph alignment
- `detectysf/utils`
  - config loading
  - random seed setup
  - metrics
- `scripts`
  - `train_detectysf.py`: end-to-end training entry
  - `build_graph.py`: rebuild news proximity graph from raw social records
- `configs`
  - default settings in `detectysf.yaml`
- `FIRST_PRINCIPLES.md`
  - project-level permanent execution principles

## Data

This project assumes data already exists at:

- `data/{dataset}_train_{16|32|64|128}.csv`
- `data/{dataset}_test.csv`
- `data/adjs/user_t5/{dataset}_nn_relations_{16|32|64|128}.pkl`
- `data/news_articles_raw/*`
- `data/social_context_raw/*`

Supported datasets: `politifact`, `gossipcop`, `fang`.

## Run

### 1) Train DetectYSF

```bash
py scripts/train_detectysf.py --config configs/detectysf.yaml --dataset_name politifact --n_shots 16
```

Optional label words:

```bash
py scripts/train_detectysf.py --config configs/detectysf.yaml --label_words real,fake
py scripts/train_detectysf.py --config configs/detectysf.yaml --label_words news,rumor
```

### 2) Rebuild graph (optional)

```bash
py scripts/build_graph.py --data_dir data --dataset_name politifact --n_shots 16 --user_threshold 5
```

## Output

Training summaries are written to:

- `logs/log_{dataset}_fewshot_{k}_samples_DetectYSF.iter{iters}.txt`

## Citation

If this project helps your research, please cite:

```text
TY  - JOUR
AU  - Jin, Weiqiang
AU  - Wang, Ningwei
AU  - Tao, Tao
AU  - Shi, Bohang
AU  - Bi, Haixia
AU  - Zhao, Biao
AU  - Wu, Hao
AU  - Duan, Haibin
AU  - Yang, Guang
PY  - 2024
DA  - 2024/08/22
TI  - A veracity dissemination consistency-based few-shot fake news detection framework by synergizing adversarial and contrastive self-supervised learning
JO  - Scientific Reports
SP  - 19470
VL  - 14
IS  - 1
SN  - 2045-2322
UR  - https://doi.org/10.1038/s41598-024-70039-9
DO  - 10.1038/s41598-024-70039-9
ID  - Jin2024
ER  -
```
