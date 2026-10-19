"""Render every sample clip as a short MP4 (log-mel spectrogram with a moving playhead, the true emotion and the
model's probabilities for all 8 emotions, with the clip's audio), plus a still thumbnail of each, so the results
can be played from the README. Writes samples/videos/ (needs ffmpeg and a trained checkpoint).

    python train.py --model wavlm
    python examples/06_render_sample_videos.py --model wavlm
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import EMOTIONS, SR
from models import MODELS, LogMel
from predict import features

BLUE, ORANGE, MUTED, INK, INK2 = "#2a78d6", "#eb6834", "#c9c7c0", "#0b0b0b", "#52514e"
FPS = 25

p = argparse.ArgumentParser()
p.add_argument("--model", default="wavlm", choices=list(MODELS))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--backbone", default="microsoft/wavlm-base-plus")
a = p.parse_args()

stem = ROOT / "results" / "checkpoints" / f"{a.model}_seed{a.seed}"
out_dir = ROOT / "samples" / "videos"
out_dir.mkdir(parents=True, exist_ok=True)
clips = sorted((ROOT / "samples").glob("*.wav"))
frontend = LogMel(augment=False).eval()


def probabilities(path):
    x = features(str(path), MODELS[a.model]["input"], a.backbone)[None]
    if Path(f"{stem}_norm.npz").exists():
        norm = np.load(f"{stem}_norm.npz")
        x = (x - norm["mean"]) / norm["std"]
    model = MODELS[a.model]["build"](x.shape)
    model.load_state_dict(torch.load(f"{stem}.pt", map_location="cpu"))
    with torch.no_grad():
        return torch.softmax(model.eval()(torch.as_tensor(x, dtype=torch.float32)), dim=1)[0].numpy()


for n, path in enumerate(clips, 1):
    true, actor = path.stem.split("_")[1], path.stem.split("_")[2].replace("actor", "")
    probs = probabilities(path)
    pred = EMOTIONS[int(probs.argmax())]
    y, _ = librosa.load(path, sr=SR)
    seconds = len(y) / SR
    mel = frontend(torch.tensor(y)[None])[0].numpy()  # spectrogram of the clip itself (no padding)

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100, facecolor="white")
    ok = "correct" if pred == true else "wrong"
    fig.text(0.03, 0.93, f"Sample {n}/{len(clips)}  ·  actor {actor}, never heard in training", fontsize=11, color=INK2)
    fig.text(0.03, 0.86, f"true emotion: {true.upper()}", fontsize=17, color=INK, weight="bold")
    fig.text(0.97, 0.86, f"{a.model} predicts: {pred.upper()} {probs.max():.0%}  ({ok})", fontsize=13,
             color=BLUE if pred == true else ORANGE, weight="bold", ha="right")
    ax = fig.add_axes((0.06, 0.38, 0.9, 0.42))
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma", extent=(0, seconds, 0, mel.shape[0]))
    ax.set_yticks([]), ax.set_xlabel("seconds", fontsize=9, color=INK2)
    ax.set_title("log-mel spectrogram", loc="left", fontsize=9, color=INK2)
    head = ax.axvline(0, color="white", lw=2)
    bars = fig.add_axes((0.06, 0.07, 0.9, 0.2))
    colors = [BLUE if e == pred else MUTED for e in EMOTIONS]
    bars.bar(range(8), probs * 100, color=colors)
    for i, v in enumerate(probs * 100):
        bars.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8, color=INK2)
    bars.set_xticks(range(8), [f"[{e}]" if e == true else e for e in EMOTIONS], fontsize=9)
    bars.set_ylim(0, 115), bars.set_yticks([])
    for side in ("top", "right", "left"):
        bars.spines[side].set_visible(False)
    bars.set_title("probability per emotion   ([ ] = true emotion)", loc="left", fontsize=9, color=INK2)

    with tempfile.TemporaryDirectory() as tmp:
        n_frames = int(np.ceil(seconds * FPS)) + FPS // 2  # half a second of hold at the end
        for f in range(n_frames):
            head.set_xdata([min(f / FPS, seconds)] * 2)
            fig.savefig(f"{tmp}/{f:04d}.png", facecolor="white")
        video = out_dir / f"{path.stem}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/%04d.png",
                        "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                        "-af", "apad", "-shortest", str(video)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{seconds / 2:.2f}", "-i", str(video),
                    "-frames:v", "1", "-vf", "scale=480:-1", "-q:v", "4", str(video.with_suffix(".jpg"))], check=True)
    plt.close(fig)
    print(f"{video.relative_to(ROOT)}   true {true:9s} predicted {pred:9s} {probs.max():.0%}")
