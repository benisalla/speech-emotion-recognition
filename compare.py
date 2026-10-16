"""Train every model (several seeds), then write results/RESULTS.md + figures (see report.py).

    python compare.py                         # all models, seeds 0 1 2
    python compare.py --models mlp mel_cnn --seeds 0
    python compare.py --report-only           # just rebuild the table/plots from results/*.json
"""
import argparse
import subprocess
import sys
from pathlib import Path

from report import ALL, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=ALL, choices=ALL)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--data", default="data/ravdess")
    p.add_argument("--out", default="results")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    out = Path(a.out)

    if not a.report_only:
        for m in a.models:
            for s in a.seeds:
                if (out / f"{m}_seed{s}.json").exists():
                    continue  # already done: makes the comparison resumable
                print(f"\n=== {m}  seed {s} ===", flush=True)
                subprocess.run([sys.executable, "train.py", "--model", m, "--seed", str(s),
                                "--data", a.data, "--out", a.out], check=True)
    report(out, a.models)


if __name__ == "__main__":
    main()
