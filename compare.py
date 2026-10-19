"""Train every model (several seeds), then write results/RESULTS.md + figures (see report.py).

    python compare.py                         # all models, seeds 0 1 2
    python compare.py --models mlp mel_cnn --seeds 0
    python compare.py --report-only           # just rebuild the table/plots from results/*.json
"""
import argparse
import subprocess
import sys
from pathlib import Path

from report import ALL, SPK, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=ALL + SPK, choices=ALL + SPK)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--data", default="data/ravdess")
    p.add_argument("--out", default="results")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--baseline", action="store_true", help="train with the anti-overfitting measures off")
    a = p.parse_args()
    out = Path(a.out)

    if not a.report_only:
        for m in a.models:
            for s in a.seeds[:1] if m.startswith("svm") else a.seeds:  # the SVM is deterministic: one run is enough
                if (out / f"{m}_seed{s}.json").exists():
                    continue  # already done: makes the comparison resumable
                print(f"\n=== {m}  seed {s} ===", flush=True)
                model, spk = m.removesuffix("_spk"), m.endswith("_spk")  # "wavlm_spk" = wavlm + --speaker-norm
                subprocess.run([sys.executable, str(Path(__file__).with_name("train.py")), "--model", model, "--seed", str(s),
                                "--data", a.data, "--out", a.out] + ["--baseline"] * a.baseline + ["--speaker-norm"] * spk,
                               check=True)
    report(out, a.models)


if __name__ == "__main__":
    main()
