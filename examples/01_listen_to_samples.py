"""Look at the 8 sample clips (one per emotion, actors never seen in training): duration, loudness, pitch, and a
log-mel spectrogram of each, saved to examples/output/samples_logmel.png.

    python examples/01_listen_to_samples.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import SR, fix_length
from models import LogMel

out = ROOT / "examples" / "output"
out.mkdir(parents=True, exist_ok=True)
clips = sorted((ROOT / "samples").glob("*.wav"))
frontend = LogMel(augment=False).eval()

fig, axes = plt.subplots(2, 4, figsize=(13, 5))
print(f"{'clip':28s} {'emotion':10s} {'sec':>4s} {'rms dB':>7s} {'median F0':>9s}")
for ax, path in zip(axes.flat, clips):
    emotion = path.stem.split("_")[1]
    y, _ = librosa.load(path, sr=SR)
    f0 = librosa.yin(y, fmin=65, fmax=500, sr=SR)
    rms = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)
    print(f"{path.name:28s} {emotion:10s} {len(y) / SR:4.1f} {rms:7.1f} {np.median(f0):7.0f} Hz")
    mel = frontend(torch.tensor(fix_length(y))[None])[0].numpy()
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(f"{emotion}  ({path.stem.split('_')[2]})", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
fig.suptitle("Log-mel spectrograms (64 bands x 3 s), what mel_cnn / crnn see", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(out / "samples_logmel.png", dpi=120)
print(f"\nsaved {out / 'samples_logmel.png'}")
