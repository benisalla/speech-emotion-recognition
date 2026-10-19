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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data import load_split, speaker_normalise
from models import BASELINE, MODELS, SPEAKER_NORM, speaker_logmel_stats, ssl_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_arrays(data_dir, kind, backbone, speaker_norm=False):
    """Returns ({split: (X, y)}, norm) for the input kind the model needs; features are standardised
    with train statistics, and norm = (mean, std) is kept so predict.py can apply the same scaling.
    With speaker_norm, feature vectors are standardised per speaker instead (see data.speaker_normalise), and
    waveform models get X = (waveform, speaker log-mel statistics) so their front-end normalises per speaker."""
    splits = {s: load_split(data_dir, s) for s in ("train", "val", "test")}
    if speaker_norm:
        assert all((d["speaker"] >= 0).all() for d in splits.values()), "--speaker-norm needs RAVDESS file names"
    if kind == "wave" and speaker_norm:
        return {s: ((d["wave"], speaker_logmel_stats(d["wave"], d["speaker"])), d["y"]) for s, d in splits.items()}, None
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
    if speaker_norm:
        return {s: (speaker_normalise(X[s], splits[s]["speaker"]), splits[s]["y"]) for s in splits}, None
    mean, std = X["train"].mean(0), X["train"].std(0) + 1e-6
    return {s: (((X[s] - mean) / std).astype(np.float32, copy=False), splits[s]["y"]) for s in splits}, (mean, std)


def speed_perturb(x, low=0.9, high=1.1):
    """Resample each clip by a random factor: shifts pitch and formants together, so the model hears a slightly
    different voice every epoch and cannot memorise the 18 training speakers. Output is cropped/padded back."""
    n, out = x.shape[1], []
    for xi in x:
        yi = F.interpolate(xi[None, None], scale_factor=float(np.random.uniform(low, high)), mode="linear")[0, 0]
        yi = yi[(len(yi) - n) // 2:][:n] if len(yi) >= n else F.pad(yi, ((n - len(yi)) // 2, n - len(yi) - (n - len(yi)) // 2))
        out.append(yi)
    return torch.stack(out)


def augment_wave(x, speed=True, max_shift=4800):
    """Speed/pitch perturbation (optional), random circular time shift (up to 0.3 s), and Gaussian noise at a
    random 15-40 dB SNR on half the batch."""
    if speed:
        x = speed_perturb(x)
    shifts = torch.randint(-max_shift, max_shift + 1, (len(x),))
    x = torch.stack([torch.roll(xi, int(s)) for xi, s in zip(x, shifts)])
    snr_db = torch.empty(len(x), 1, device=x.device).uniform_(15, 40)
    noise = torch.randn_like(x) * x.std(1, keepdim=True) / 10 ** (snr_db / 20)
    return x + noise * (torch.rand(len(x), 1, device=x.device) < 0.5)


def rows(X, idx):
    """Rows idx of X, an array or a tuple of arrays (waveform, speaker statistics)."""
    return tuple(x[idx] for x in X) if isinstance(X, tuple) else X[idx]


def tensors(X):
    """The model inputs of X as a tuple of tensors on DEVICE."""
    return tuple(torch.as_tensor(x, device=DEVICE) for x in (X if isinstance(X, tuple) else (X,)))


@torch.no_grad()
def logits(model, X, batch_size=64):
    model.eval()
    n = len(X[0]) if isinstance(X, tuple) else len(X)
    return torch.cat([model(*tensors(rows(X, slice(i, i + batch_size)))).cpu() for i in range(0, n, batch_size)])


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


def mixup(x, y, alpha):
    """Mixup: train on convex combinations of pairs of examples; returns mixed x, both targets and the weight."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(x), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


def train_torch(name, data, epochs, lr, patience, batch_size, seed, ckpt_path, baseline=False):
    """Anti-overfitting measures (per-model strength in models.MODELS): AdamW weight decay, mixup and Gaussian noise
    on feature vectors, speed/pitch perturbation of waveforms, and an exponential moving average (EMA) of the weights, which is the
    model that gets validated, checkpointed and tested. Train accuracy/loss are also measured the honest way, in eval
    mode on clean (un-augmented) train clips, so the logged train-vs-val gap is the real generalisation gap."""
    cfg = (BASELINE if baseline else MODELS)[name]
    (Xtr, ytr), (Xv, yv) = data["train"], data["val"]
    n_train = len(ytr)
    model = cfg["build"]((Xtr[0] if isinstance(Xtr, tuple) else Xtr).shape).to(DEVICE)
    steps = epochs * (n_train // batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    ema = (torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg["ema"]),
                                               use_buffers=True) if cfg["ema"] else None)
    evaluated = lambda: ema.module if ema else model  # the weights we validate / keep
    ytr_t = torch.as_tensor(ytr)
    g = torch.Generator().manual_seed(seed)
    probe = torch.randperm(n_train, generator=torch.Generator().manual_seed(0))[:240].numpy()  # fixed clean-train subset

    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(epochs):
        model.train()
        perm, losses = torch.randperm(n_train, generator=g), []
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size].numpy()
            (xb, *extra), yb = tensors(rows(Xtr, idx)), ytr_t[idx].to(DEVICE)  # extra = speaker statistics, if any
            if cfg["input"] == "wave":
                xb = augment_wave(xb, speed=cfg["speed"])
            elif cfg["noise"]:
                xb = xb + cfg["noise"] * torch.randn_like(xb)
            if cfg["mixup"]:
                xb, ya, yb2, lam = mixup(xb, yb, cfg["mixup"])
                out = model(xb)
                loss = lam * loss_fn(out, ya) + (1 - lam) * loss_fn(out, yb2)
            else:
                loss = loss_fn(model(xb, *extra), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if ema:
                ema.update_parameters(model)
            losses.append(loss.item())
        net = evaluated()
        tr_logits, val_logits = logits(net, rows(Xtr, probe)), logits(net, Xv)
        val = scores(yv, val_logits.argmax(1).numpy())
        history.append(dict(epoch=epoch, lr=sched.get_last_lr()[0], fit_loss=float(np.mean(losses)),
                            train_loss=loss_fn(tr_logits, torch.as_tensor(ytr[probe])).item(),
                            train_acc=float((tr_logits.argmax(1).numpy() == ytr[probe]).mean()),
                            val_loss=loss_fn(val_logits, torch.as_tensor(yv)).item(),
                            val_acc=val["acc"], val_f1=val["f1"]))
        if val["f1"] > best_f1:
            best_f1, best_epoch = val["f1"], epoch
            torch.save(net.state_dict(), ckpt_path)
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
    p.add_argument("--baseline", action="store_true", help="same model and schedule with the anti-overfitting measures off")
    p.add_argument("--speaker-norm", action="store_true",
                   help="normalise features per speaker; uses unlabelled clips of each test speaker")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = (BASELINE if a.baseline else MODELS).get(a.model, dict(input="feats"))
    if a.speaker_norm and a.model not in SPEAKER_NORM:
        p.error(f"--speaker-norm applies to: {', '.join(SPEAKER_NORM)}")
    name = f"{a.model}_spk" if a.speaker_norm else a.model  # results / checkpoint name
    data, norm = get_arrays(a.data, cfg["input"], a.backbone, a.speaker_norm)
    if norm is not None and a.model != "svm":  # predict.py needs the scaling of torch feature models
        np.savez(out / "checkpoints" / f"{name}_seed{a.seed}_norm.npz", mean=norm[0], std=norm[1])

    t0 = time.time()
    if a.model == "svm":
        predict_fn, info = train_svm(data)
    else:
        predict_fn, info = train_torch(a.model, data, a.epochs or cfg["epochs"], a.lr or cfg["lr"], cfg["patience"],
                                       a.batch_size, a.seed, out / "checkpoints" / f"{name}_seed{a.seed}.pt", a.baseline)
    seconds = time.time() - t0

    (Xv, yv), (Xte, yte) = data["val"], data["test"]
    test_pred = predict_fn(Xte)
    result = dict(model=name, input=cfg["input"], seed=a.seed, recipe="baseline" if a.baseline else "regularised",
                  train_seconds=round(seconds, 1), **info,
                  val=scores(yv, predict_fn(Xv)), test=scores(yte, test_pred),
                  confusion=confusion_matrix(yte, test_pred, labels=range(8)).tolist())
    (out / f"{name}_seed{a.seed}.json").write_text(json.dumps(result, indent=2))
    print(f"\n{name}: test acc {result['test']['acc']:.3f}  macro-F1 {result['test']['f1']:.3f}  ({seconds:.0f}s)")


if __name__ == "__main__":
    main()
