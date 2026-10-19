"""Overfitting, before and after: the same MLP trained with the original recipe and with the regularised one
(input dropout, smaller layers, stronger weight decay, mixup, feature noise, EMA weights). Prints the clean train
accuracy vs validation accuracy (the generalisation gap) and the test score of both. About 2-3 min on a CPU.

    python examples/03_overfitting_fix.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from train import get_arrays, scores, train_torch

out = ROOT / "examples" / "output"
out.mkdir(parents=True, exist_ok=True)
data, _ = get_arrays(str(ROOT / "data" / "ravdess"), "feats", backbone=None)
Xte, yte = data["test"]

for baseline in (True, False):
    torch.manual_seed(0), np.random.seed(0)
    name = "no-regularisation" if baseline else "regularised      "
    predict, info = train_torch("mlp", data, epochs=300, lr=1e-3, patience=40, batch_size=32, seed=0,
                                ckpt_path=out / f"mlp_{name.strip()}.pt", baseline=baseline)
    h = info["history"][info["best_epoch"]]
    print(f"\n>>> {name}  train acc {h['train_acc']:.1%}  val acc {h['val_acc']:.1%}  "
          f"gap {h['train_acc'] - h['val_acc']:+.1%}  test acc {scores(yte, predict(Xte))['acc']:.1%}\n")
