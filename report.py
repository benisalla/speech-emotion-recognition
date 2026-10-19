"""Build results/RESULTS.md and every figure from the results/*.json files written by train.py.

    python report.py        # (compare.py also calls this at the end)

Figures:
    accuracy_comparison.png   test accuracy + macro-F1 per model (mean ± std over seeds)
    training_curves.png       train vs val loss and accuracy per epoch, for every neural model
    confusion_matrices.png    row-normalised test confusion matrix of every model
    per_class_f1.png          test F1 per emotion x model
    overfitting.png           with vs without anti-overfitting measures: train-test gap, test accuracy (results/baseline/)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import EMOTIONS

ALL = ["notebook_fnn", "svm", "mlp", "wave_cnn", "mfcc_cnn", "mel_cnn", "crnn", "wavlm"]
SPK = ["svm_spk", "mlp_spk", "mfcc_cnn_spk", "mel_cnn_spk", "crnn_spk", "wavlm_spk"]  # train.py --speaker-norm
INPUT = {"notebook_fnn": "notebook features (199-d)", "svm": "handcrafted features", "mlp": "handcrafted features",
         "wave_cnn": "raw waveform", "mfcc_cnn": "MFCC sequence", "mel_cnn": "log-mel spectrogram",
         "crnn": "log-mel spectrogram", "wavlm": "frozen WavLM states"}
INPUT |= {m: f"{INPUT[m.removesuffix('_spk')]}, speaker-normalised" for m in SPK}

BLUE, ORANGE = "#2a78d6", "#eb6834"                 # series 1 / series 2
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8, "axes.axisbelow": True, "ytick.labelcolor": INK2,
    "xtick.labelcolor": INK2,
})


def load(out, models):
    rows = []
    for m in models:
        runs = [json.loads(p.read_text()) for p in sorted(Path(out).glob(f"{m}_seed*.json"))]
        if runs:
            acc, f1 = (np.array([r["test"][k] for r in runs]) * 100 for k in ("acc", "f1"))
            rows.append(dict(model=m, runs=runs, acc=acc, f1=f1))
    return sorted(rows, key=lambda r: r["acc"].mean())  # worst -> best (bottom -> top in bar charts)


def sd(x):
    """Sample standard deviation over seeds (0 for a single run)."""
    return x.std(ddof=1) if len(x) > 1 else 0.0


def table(rows):
    lines = ["| model | input | test accuracy | test macro-F1 | params | train time / run | seeds |",
             "|---|---|---|---|---|---|---|"]
    for r in reversed(rows):
        runs = r["runs"]
        params = f"{runs[0]['params'] / 1e6:.2f} M" if "params" in runs[0] else "–"
        lines.append(f"| `{r['model']}` | {INPUT[r['model']]} | **{r['acc'].mean():.1f}** ± {sd(r['acc']):.1f} % "
                     f"| {r['f1'].mean():.1f} ± {sd(r['f1']):.1f} % | {params} "
                     f"| {np.mean([x['train_seconds'] for x in runs]):.0f} s | {len(runs)} |")
    return "\n".join(lines)


def plot_comparison(rows, path):
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(rows) + 1.4))
    y = np.arange(len(rows))
    for off, key, label, color in ((0.2, "acc", "accuracy", BLUE), (-0.2, "f1", "macro-F1", ORANGE)):
        means, stds = [r[key].mean() for r in rows], [sd(r[key]) for r in rows]
        ax.barh(y + off, means, 0.38, xerr=stds, color=color, label=label,
                error_kw=dict(ecolor=INK2, elinewidth=1, capsize=0))
        for yi, m, s in zip(y + off, means, stds):
            ax.text(m + s + 1, yi, f"{m:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(100 / 8, ls="--", c=MUTED, lw=1)
    ax.set_ylim(-1.0, len(rows) - 0.5)
    ax.text(100 / 8 + 0.8, -0.85, "chance (12.5%)", color=MUTED, fontsize=7.5, va="bottom")
    ax.set_yticks(y, [f"{r['model']}\n{INPUT[r['model']]}" for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 100)
    ax.set_xlabel("test score (%), mean ± std over seeds")
    ax.legend(loc="lower right")
    ax.set_title("Test accuracy and macro-F1 per model", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _mean_curve(runs, key):
    """Mean and min/max over seeds of a per-epoch metric; runs stop at different epochs (early stopping)."""
    n = max(len(r["history"]) for r in runs)
    a = np.full((len(runs), n), np.nan)
    for i, r in enumerate(runs):
        a[i, :len(r["history"])] = [h[key] for h in r["history"]]
    return np.nanmean(a, 0), np.nanmin(a, 0), np.nanmax(a, 0)


def plot_training_curves(rows, path):
    rows = [r for r in reversed(rows) if "history" in r["runs"][0]]
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 1.9 * len(rows) + 0.6), squeeze=False)
    for (ax_loss, ax_acc), r in zip(axes, rows):
        for ax, metric in ((ax_loss, "loss"), (ax_acc, "acc")):
            for split, color in (("train", BLUE), ("val", ORANGE)):
                mean, lo, hi = _mean_curve(r["runs"], f"{split}_{metric}")
                x = np.arange(len(mean))
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
                ax.plot(x, mean, color=color, lw=1.6, label=split)
            for b in {run["best_epoch"] for run in r["runs"]}:
                ax.axvline(b, color=MUTED, lw=0.8, ls=":")
        ax_loss.set_ylabel(r["model"], color=INK, fontsize=9)
        ax_acc.set_ylim(0, 1)
    axes[0, 0].set_title("loss (label-smoothed cross-entropy)", loc="left")
    axes[0, 1].set_title("accuracy (train: clean clips, eval mode)", loc="left")
    axes[0, 1].legend(loc="lower right", ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    fig.suptitle("Training curves: mean over seeds, band = min/max, dotted = best val epoch (checkpoint kept)",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _confusion(r):
    """Test confusion matrix summed over seeds."""
    return np.sum([run["confusion"] for run in r["runs"]], axis=0).astype(float)


def plot_confusions(rows, path):
    rows = list(reversed(rows))
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(rows) / cols)), cols, figsize=(3.1 * cols, 3.2 * np.ceil(len(rows) / cols)),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, rows):
        ax.axis("on")
        ax.grid(False)
        cm = _confusion(r)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(8):
            for j in range(8):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(8), [e[:3] for e in EMOTIONS], fontsize=7, rotation=90)
        ax.set_yticks(range(8), EMOTIONS, fontsize=7)
        ax.set_title(f"{r['model']}  ({r['acc'].mean():.1f}%)", fontsize=9)
    fig.suptitle("Test confusion matrices (rows = true emotion, % of row, summed over seeds)",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(rows, path):
    rows = list(reversed(rows))
    f1 = []
    for r in rows:
        cm = _confusion(r)
        tp = np.diag(cm)
        f1.append(2 * tp / (cm.sum(0) + cm.sum(1)) * 100)
    f1 = np.array(f1)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.3))
    ax.imshow(f1, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.grid(False)
    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            ax.text(j, i, f"{f1[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if f1[i, j] > 55 else INK)
    ax.set_xticks(range(8), EMOTIONS)
    ax.set_yticks(range(len(rows)), [r["model"] for r in rows])
    ax.tick_params(length=0)
    ax.set_title("Test F1 per emotion (%), models sorted best -> worst", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _at_best(run, key):
    return run["history"][run["best_epoch"]][key] * 100


def overfitting_rows(out, models):
    """Per model, no regularisation (results/baseline) vs regularised (+ speaker-normalised when it exists):
    clean train acc, val acc and test acc at the kept checkpoint, mean over seeds."""
    rows = []
    for m in models:
        row = dict(model=m)
        for recipe, folder, name in (("baseline", Path(out) / "baseline", m), ("regularised", Path(out), m),
                                     ("speaker_norm", Path(out), f"{m}_spk")):
            runs = [json.loads(p.read_text()) for p in sorted(folder.glob(f"{name}_seed*.json"))]
            if runs and "history" in runs[0]:  # the SVM has no epochs
                row[recipe] = {k: np.mean([_at_best(r, f"{k}_acc") for r in runs]) for k in ("train", "val")}
                row[recipe]["test"] = np.mean([r["test"]["acc"] * 100 for r in runs])
        if "baseline" in row and "regularised" in row:
            rows.append(row)
    return rows


RECIPES = (("baseline", "no regularisation", MUTED), ("regularised", "regularised", BLUE),
           ("speaker_norm", "regularised + speaker norm", ORANGE))


def overfitting_table(rows):
    lines = ["| model | recipe | train acc | val acc | test acc | gap (train - test) |", "|---|---|---|---|---|---|"]
    for r in rows:
        for recipe, name, _ in RECIPES:
            if recipe not in r:
                continue
            d = r[recipe]
            lines.append(f"| `{r['model']}` | {name} | {d['train']:.1f} % | {d['val']:.1f} % | {d['test']:.1f} % "
                         f"| {d['train'] - d['test']:.1f} |")
    return "\n".join(lines)


def plot_overfitting(rows, path):
    fig, (ax_gap, ax_test) = plt.subplots(1, 2, figsize=(10, 0.75 * len(rows) + 1.6), sharey=True)
    y = np.arange(len(rows))
    for off, (recipe, label, color) in zip((-0.27, 0, 0.27), RECIPES):
        sub = [(yi, r[recipe]) for yi, r in zip(y, rows) if recipe in r]
        if not sub:
            continue
        ys = np.array([yi for yi, _ in sub]) + off
        gap = [d["train"] - d["test"] for _, d in sub]
        test = [d["test"] for _, d in sub]
        ax_gap.barh(ys, gap, 0.26, color=color, label=label)
        ax_test.barh(ys, test, 0.26, color=color, label=label)
        for yi, g, t in zip(ys, gap, test):
            ax_gap.text(g + 0.5, yi, f"{g:.0f}", va="center", fontsize=7.5, color=INK2)
            ax_test.text(t + 0.5, yi, f"{t:.1f}", va="center", fontsize=7.5, color=INK2)
    ax_gap.set_yticks(y, [r["model"] for r in rows])
    ax_gap.invert_yaxis()
    for ax in (ax_gap, ax_test):
        ax.grid(axis="y", visible=False)
    ax_gap.set_title("overfitting gap: train acc - test acc (points)", loc="left")
    ax_test.set_title("test accuracy on unseen actors (%)", loc="left")
    ax_test.set_xlim(0, 100)
    fig.legend(*ax_test.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, fontsize=7.5)
    fig.suptitle("Without vs with anti-overfitting measures, same schedule (mean over seeds; train acc on clean clips)",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(out="results", models=ALL):
    out = Path(out)
    rows = load(out, models)
    md = table(rows)
    text = "# Results\n\nTest-set scores, mean ± sample std over seeds (generated by `report.py`).\n\n" + md + "\n"
    print(md)
    over = overfitting_rows(out, models)
    if over:
        om = overfitting_table(over)
        text += "\n## Overfitting: without vs with the anti-overfitting measures\n\n" + om + "\n"
        print(om)
        plot_overfitting(over, out / "overfitting.png")
    (out / "RESULTS.md").write_text(text)
    plot_comparison(rows, out / "accuracy_comparison.png")
    plot_training_curves(rows, out / "training_curves.png")
    plot_confusions(rows, out / "confusion_matrices.png")
    plot_per_class_f1(rows, out / "per_class_f1.png")


if __name__ == "__main__":
    report(models=ALL + SPK)
