"""Classify the 8 sample clips with a trained checkpoint and compare with their true emotion.

    python train.py --model wavlm            # or use the checkpoint of any other model
    python examples/05_classify_samples.py --model wavlm
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--model", default="wavlm")
a = p.parse_args()

hits = 0
clips = sorted((ROOT / "samples").glob("*.wav"))
for path in clips:
    true = path.stem.split("_")[1]
    top3 = subprocess.run([sys.executable, str(ROOT / "predict.py"), str(path), "--model", a.model,
                           "--ckpt-dir", str(ROOT / "results" / "checkpoints")],
                          capture_output=True, text=True, check=True).stdout.split("\n")[:3]
    pred = top3[0].split()[0]
    hits += pred == true
    print(f"{path.name:28s} true {true:10s} predicted {pred:10s} {'ok' if pred == true else '--'}   "
          f"top-3: {' | '.join(' '.join(t.split()) for t in top3)}")
print(f"\n{hits}/{len(clips)} correct")
