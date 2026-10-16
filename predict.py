"""Predict the emotion of one .wav file with a trained checkpoint.

    python predict.py path/to/clip.wav --model wavlm
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from data import EMOTIONS, fix_length, handcrafted_features, load_trimmed, notebook_features
from models import MODELS, ssl_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--model", default="mel_cnn", choices=list(MODELS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backbone", default="microsoft/wavlm-base-plus")
    p.add_argument("--ckpt-dir", default="results/checkpoints")
    a = p.parse_args()

    kind = MODELS[a.model]["input"]
    y = load_trimmed(a.wav)
    x = {"wave": lambda: fix_length(y),
         "feats": lambda: handcrafted_features(y),
         "nb_feats": lambda: notebook_features(a.wav),
         "ssl": lambda: ssl_features(fix_length(y)[None], a.backbone)[0]}[kind]()[None]

    stem = Path(a.ckpt_dir) / f"{a.model}_seed{a.seed}"
    if Path(f"{stem}_norm.npz").exists():
        norm = np.load(f"{stem}_norm.npz")
        x = (x - norm["mean"]) / norm["std"]
    model = MODELS[a.model]["build"](x.shape)
    model.load_state_dict(torch.load(f"{stem}.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        probs = torch.softmax(model(torch.as_tensor(x, dtype=torch.float32)), dim=1)[0]
    for i in probs.argsort(descending=True)[:3]:
        print(f"{EMOTIONS[i]:10s} {probs[i]:.1%}")


if __name__ == "__main__":
    main()
