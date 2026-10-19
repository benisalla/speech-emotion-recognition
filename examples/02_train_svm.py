"""The quickest real result: an RBF SVM on 258 handcrafted features, trained on actors 1-18 and tested on the
unseen actors 21-24. Runs in seconds once the features are cached (first run: ~2 min of preprocessing).

    python prepare_ravdess.py   # once
    python examples/02_train_svm.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import classification_report
from sklearn.svm import SVC

from data import EMOTIONS
from train import get_arrays

data, _ = get_arrays(str(ROOT / "data" / "ravdess"), "feats", backbone=None)
(Xtr, ytr), (Xte, yte) = data["train"], data["test"]
clf = SVC(C=3, gamma="scale").fit(Xtr, ytr)
print(f"train accuracy {clf.score(Xtr, ytr):.1%}   test accuracy (unseen actors) {clf.score(Xte, yte):.1%}\n")
print(classification_report(yte, clf.predict(Xte), target_names=EMOTIONS, digits=3))
