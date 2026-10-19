"""Predict the emotion of one .wav file with a trained checkpoint.

    python predict.py path/to/clip.wav --model wavlm
    python predict.py path/to/clip.wav --model wavlm_spk --speaker-ref path/to/same_speaker_clips/

Needs a checkpoint from train.py (results/checkpoints/). A head trained with --backbone X
must be used with the same --backbone X here. Speaker-normalised models (`*_spk`) also need a few
other clips of the same speaker (any emotion, no labels) to compute that speaker's feature statistics.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from data import EMOTIONS, fix_length, handcrafted_features, load_trimmed, notebook_features
from models import MODELS, SPEAKER_NORM, speaker_logmel_stats, ssl_features


def features(path, kind, backbone):
    y = load_trimmed(path)
    return {"wave": lambda: fix_length(y),
            "feats": lambda: handcrafted_features(y),
            "nb_feats": lambda: notebook_features(path),
            "ssl": lambda: ssl_features(fix_length(y)[None], backbone)[0]}[kind]()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--model", default="mel_cnn", choices=[*MODELS, *(f"{m}_spk" for m in SPEAKER_NORM if m != "svm")])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backbone", default="microsoft/wavlm-base-plus")
    p.add_argument("--ckpt-dir", default="results/checkpoints")
    p.add_argument("--speaker-ref", help="folder of other .wav clips of the same speaker (for *_spk models)")
    a = p.parse_args()

    stem = Path(a.ckpt_dir) / f"{a.model}_seed{a.seed}"
    base = a.model.removesuffix("_spk")
    if not Path(f"{stem}.pt").exists():
        flag = " --speaker-norm" if a.model.endswith("_spk") else ""
        raise SystemExit(f"no checkpoint at {stem}.pt - train it first: python train.py --model {base}{flag} --seed {a.seed}")
    kind = MODELS[base]["input"]
    x = features(a.wav, kind, a.backbone)[None]

    inputs = []
    if a.model.endswith("_spk"):
        refs = sorted(Path(a.speaker_ref).glob("*.wav")) if a.speaker_ref else []
        if len(refs) < 5:
            raise SystemExit("speaker-normalised models need --speaker-ref with at least 5 clips of the same speaker")
        ref = np.stack([features(str(r), kind, a.backbone) for r in refs] + [x[0]])
        if kind == "wave":  # the log-mel front-end gets the speaker's per-band statistics
            inputs = [torch.as_tensor(speaker_logmel_stats(ref, np.zeros(len(ref)))[:1])]
        else:
            x = (x - ref.mean(0)) / (ref.std(0) + 1e-6)
    elif Path(f"{stem}_norm.npz").exists():
        norm = np.load(f"{stem}_norm.npz")
        x = (x - norm["mean"]) / norm["std"]
    model = MODELS[base]["build"](x.shape)
    model.load_state_dict(torch.load(f"{stem}.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        probs = torch.softmax(model(torch.as_tensor(x, dtype=torch.float32), *inputs), dim=1)[0]
    for i in probs.argsort(descending=True)[:3]:
        print(f"{EMOTIONS[i]:10s} {probs[i]:.1%}")


if __name__ == "__main__":
    main()
