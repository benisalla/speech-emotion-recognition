"""Train + evaluate one model.

    python train.py --model mel_cnn
    python train.py --model svm --data data/ravdess
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data import load_split
from models import MODELS, ssl_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_arrays(data_dir, kind, backbone):
    """Returns ({split: (X, y)}, norm) for the input kind the model needs; features are standardised
    with train statistics, and norm = (mean, std) is kept so predict.py can apply the same scaling."""
    splits = {s: load_split(data_dir, s) for s in ("train", "val", "test")}
    if kind in ("wave", "nb_feats"):  # raw audio / notebook features (already min-max scaled per clip)
        return {s: (d[kind], d["y"]) for s, d in splits.items()}, None
    if kind == "ssl":
        X = {}
        for s, d in splits.items():
            cache = Path("cache") / f"{Path(data_dir).name}_{s}_{backbone.replace('/', '_')}.npy"
            if not cache.exists():
                np.save(cache, ssl_features(d["wave"], backbone))
            X[s] = np.load(cache)
    else:  # handcrafted feats
        X = {s: d["feats"] for s, d in splits.items()}
    mean, std = X["train"].mean(0), X["train"].std(0) + 1e-6
    return {s: (((X[s] - mean) / std).astype(np.float32, copy=False), splits[s]["y"]) for s in splits}, (mean, std)


def augment_wave(x, max_shift=4800):
    """Random circular time shift (up to 0.3 s) + Gaussian noise at a random 15-40 dB SNR on half the batch."""
    shifts = torch.randint(-max_shift, max_shift + 1, (len(x),))
    x = torch.stack([torch.roll(xi, int(s)) for xi, s in zip(x, shifts)])
    snr_db = torch.empty(len(x), 1, device=x.device).uniform_(15, 40)
    noise = torch.randn_like(x) * x.std(1, keepdim=True) / 10 ** (snr_db / 20)
    return x + noise * (torch.rand(len(x), 1, device=x.device) < 0.5)


@torch.no_grad()
def logits(model, X, batch_size=64):
    model.eval()
    return torch.cat([model(torch.as_tensor(X[i:i + batch_size], device=DEVICE)).cpu()
                      for i in range(0, len(X), batch_size)])


def predict(model, X):
    return logits(model, X).argmax(1).numpy()


def scores(y, pred):
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, average="macro"))


def train_svm(data):
    from sklearn.svm import SVC
    (Xtr, ytr), (Xv, yv) = data["train"], data["val"]
    best = max((scores(yv, SVC(C=C, gamma="scale").fit(Xtr, ytr).predict(Xv))["f1"], C) for C in (0.3, 1, 3, 10, 30))
    clf = SVC(C=best[1], gamma="scale").fit(Xtr, ytr)
    return clf.predict, dict(C=best[1], n_support_vectors=int(len(clf.support_vectors_)))


def train_torch(name, data, epochs, lr, patience, batch_size, seed, ckpt_path):
    cfg = MODELS[name]
    (Xtr, ytr), (Xv, yv) = data["train"], data["val"]
    model = cfg["build"](Xtr.shape).to(DEVICE)
    steps = epochs * (len(Xtr) // batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xtr_t, ytr_t = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    g = torch.Generator().manual_seed(seed)

    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(epochs):
        model.train()
        perm, losses, correct, seen = torch.randperm(len(Xtr), generator=g), [], 0, 0
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr_t[idx].to(DEVICE), ytr_t[idx].to(DEVICE)
            if cfg["input"] == "wave":
                xb = augment_wave(xb)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            correct, seen = correct + (out.argmax(1) == yb).sum().item(), seen + len(yb)
        val_logits = logits(model, Xv)
        val = scores(yv, val_logits.argmax(1).numpy())
        history.append(dict(epoch=epoch, lr=sched.get_last_lr()[0], train_loss=float(np.mean(losses)),
                            train_acc=correct / seen, val_loss=loss_fn(val_logits, torch.as_tensor(yv)).item(),
                            val_acc=val["acc"], val_f1=val["f1"]))
        if val["f1"] > best_f1:
            best_f1, best_epoch = val["f1"], epoch
            torch.save(model.state_dict(), ckpt_path)
        h = history[-1]
        print(f"epoch {epoch:3d}  train loss {h['train_loss']:.3f} acc {h['train_acc']:.3f}  "
              f"val loss {h['val_loss']:.3f} acc {h['val_acc']:.3f} f1 {h['val_f1']:.3f}"
              f"{'  *' if best_epoch == epoch else ''}", flush=True)
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return (lambda X: predict(model, X)), dict(best_epoch=best_epoch, params=sum(p.numel() for p in model.parameters()),
                                               history=history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["svm", *MODELS])
    p.add_argument("--data", default="data/ravdess")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backbone", default="microsoft/wavlm-base-plus", help="any HF wav2vec2/HuBERT/WavLM model")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = MODELS.get(a.model, dict(input="feats"))
    data, norm = get_arrays(a.data, cfg["input"], a.backbone)
    if norm is not None:
        np.savez(out / "checkpoints" / f"{a.model}_seed{a.seed}_norm.npz", mean=norm[0], std=norm[1])

    t0 = time.time()
    if a.model == "svm":
        predict_fn, info = train_svm(data)
    else:
        predict_fn, info = train_torch(a.model, data, a.epochs or cfg["epochs"], a.lr or cfg["lr"], cfg["patience"],
                                       a.batch_size, a.seed, out / "checkpoints" / f"{a.model}_seed{a.seed}.pt")
    seconds = time.time() - t0

    (Xv, yv), (Xte, yte) = data["val"], data["test"]
    test_pred = predict_fn(Xte)
    result = dict(model=a.model, input=cfg["input"], seed=a.seed, train_seconds=round(seconds, 1), **info,
                  val=scores(yv, predict_fn(Xv)), test=scores(yte, test_pred),
                  confusion=confusion_matrix(yte, test_pred, labels=range(8)).tolist())
    (out / f"{a.model}_seed{a.seed}.json").write_text(json.dumps(result, indent=2))
    print(f"\n{a.model}: test acc {result['test']['acc']:.3f}  macro-F1 {result['test']['f1']:.3f}  ({seconds:.0f}s)")


if __name__ == "__main__":
    main()
