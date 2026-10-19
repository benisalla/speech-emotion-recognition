"""Why the models overfit: they memorise the 18 training voices. Standardising features per speaker removes most of
the voice identity and leaves the emotion, so the same SVM jumps by about 12 points on unseen actors.
(It uses a few unlabelled clips of each test speaker, so it is reported separately from the main protocol.)

    python examples/04_speaker_normalisation.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.svm import SVC

from train import get_arrays, scores

for speaker_norm in (False, True):
    data, _ = get_arrays(str(ROOT / "data" / "ravdess"), "feats", backbone=None, speaker_norm=speaker_norm)
    (Xtr, ytr), (Xte, yte) = data["train"], data["test"]
    s = scores(yte, SVC(C=3, gamma="scale").fit(Xtr, ytr).predict(Xte))
    print(f"{'per-speaker' if speaker_norm else 'global     '} normalisation:  "
          f"test accuracy {s['acc']:.1%}  macro-F1 {s['f1']:.1%}")
