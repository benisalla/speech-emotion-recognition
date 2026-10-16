# Speech Emotion Recognition: 8 audio classifiers compared

> Detect emotion in speech, and find out which way of feeding audio to a model works best:
> raw waveform, MFCCs, mel-spectrograms, handcrafted features or a pretrained speech model.
> 7 neural models + an SVM baseline, all trained on the same data and split, and evaluated on
> **speakers never seen during training**.

**Keywords:** speech emotion recognition (SER) · audio classification · emotion detection · PyTorch · RAVDESS ·
WavLM · wav2vec2 · self-supervised learning · mel-spectrogram · MFCC · CNN · CRNN · raw waveform · SVM ·
librosa · deep learning · speaker-independent evaluation

![Test accuracy and macro-F1 per model](results/accuracy_comparison.png)

## Contents

1. [Results](#results)
2. [Figures](#figures)
3. [Models](#models)
4. [Training and evaluation setup](#training-and-evaluation-setup)
5. [Quick start](#quick-start)
6. [Usage](#usage)
7. [Repository layout](#repository-layout)
8. [Dataset](#dataset)

## Results

RAVDESS speech, 8 emotions, speaker-independent test set (actors 21-24, 240 clips).
Mean ± sample std over 3 seeds. Chance = 12.5 %.

| model | input | test accuracy | test macro-F1 | params | train time / run ¹ |
|---|---|---|---|---|---|
| `wavlm` | frozen WavLM states | **77.6** ± 1.6 % | 76.7 ± 1.5 % | 0.40 M ² | 52 s ³ |
| `crnn` | log-mel spectrogram | **62.8** ± 0.5 % | 60.8 ± 1.4 % | 0.32 M | 426 s |
| `mel_cnn` | log-mel spectrogram | **57.1** ± 1.8 % | 48.9 ± 4.2 % | 0.36 M | 445 s |
| `mfcc_cnn` | MFCC sequence | **55.6** ± 0.6 % | 54.5 ± 0.9 % | 0.17 M | 118 s |
| `svm` ⁴ | handcrafted features | **49.2** % | 48.0 % | – | 1 s |
| `mlp` | handcrafted features | **47.4** ± 1.0 % | 46.4 ± 1.2 % | 0.27 M | 212 s |
| `wave_cnn` | raw waveform | **39.4** ± 2.7 % | 36.6 ± 3.4 % | 0.15 M | 137 s |
| `notebook_fnn` | notebook features (199-d) | **27.8** ± 4.2 % | 24.0 ± 3.9 % | 1.48 M | 77 s |

¹ On a shared 4-core laptop CPU, no GPU.
² Trainable head only; the frozen WavLM backbone has 94 M parameters.
³ Excludes the one-off WavLM feature extraction (about 13 min on CPU, cached afterwards).
⁴ The SVM is deterministic, so it is run once.

The same table is generated in [`results/RESULTS.md`](results/RESULTS.md).

### Key findings

- **A pretrained speech model wins by a wide margin.** A frozen WavLM plus a small head reaches 77.6 %, 2.8× the
  original notebook pipeline (27.8 %). With only 1,080 training clips, representations learned from large amounts of
  unlabelled speech beat every model trained from scratch.
- **Among models trained from scratch, time-frequency inputs beat raw audio.** The spectrogram and MFCC models
  (55.6-62.8 %) clearly beat the raw-waveform CNN (39.4 %), which has to learn its own filterbank from little data.
- **Modelling time helps.** The CRNN (CNN + BiGRU + attention) beats the plain 2D CNN on the same log-mel input,
  by +5.7 points accuracy and +11.9 points macro-F1.
- **The feature pipeline matters.** The same feature families (MFCC, chroma, mel, contrast, tonnetz) go from 27.8 %
  with the notebook pipeline to 47.4-49.2 % with a better one: 16 kHz audio with silence trimmed, log-mel,
  mean + std pooling and standardisation on train statistics.
- **Hardest emotions for the best models (`wavlm`, `crnn`):** *sad* is often taken for *calm*, and *happy* for
  *surprised*. Weaker models struggle most with *neutral* (`mel_cnn`, `wave_cnn`, `notebook_fnn`) or *happy*
  (`svm`, `mlp`). See the per-class F1 figure.

> Many published RAVDESS results (80-90 %+) use **random** splits, where the same actors appear in train and test.
> This repo tests on unseen actors, which is harder and closer to real use.

## Figures

All figures are regenerated from `results/*.json` by `python report.py`.

### Training curves

Train vs validation loss and accuracy per epoch, for every neural model. The line is the mean over seeds, the band is
the min/max, and the dotted lines mark the epoch kept by early stopping (best validation macro-F1).

![Training curves](results/training_curves.png)

How to read them:

- `mlp` overfits early: validation loss is lowest at epochs 15-17, while train accuracy climbs to 98 %.
- `wavlm` passes 95 % train accuracy by about epoch 24. Its validation accuracy stays roughly flat after that.
- `notebook_fnn` underfits: at the original learning rate of 1e-4, train accuracy is still only 44-53 % when early
  stopping ends training (after 150-240 epochs).
- The four CNNs keep their best checkpoint late in the 40-epoch budget (epochs 22-38), and their train accuracy is
  still well below 100 % (58-89 %). They are probably under-trained, so a longer schedule may help
  (see [Usage](#usage)). Part of the late gain comes from the one-cycle schedule lowering the learning rate at the end.

### Per-emotion F1

![Per-class F1](results/per_class_f1.png)

### Confusion matrices

![Confusion matrices](results/confusion_matrices.png)

## Models

One model per way of representing audio:

| model | what the model sees | architecture |
|---|---|---|
| `notebook_fnn` | 199 features from the original notebook: MFCC, chroma, mel, contrast and tonnetz averaged over the clip, min-max scaled per clip | the original Colab network: 6 dense ReLU layers (1000-750-500-250-100-50) |
| `svm` | 258 handcrafted features: 129 per-frame features (40 MFCC, 12 chroma, 64 log-mel, 7 contrast, 6 tonnetz), pooled with mean **and** std and standardised | RBF-kernel SVM, `C` tuned on validation |
| `mlp` | the same 258 features | 2-layer MLP (512-256) with BatchNorm + dropout |
| `wave_cnn` | **raw waveform** (3 s at 16 kHz = 48,000 samples) | 1D CNN; a wide strided first conv learns its own filterbank (M5-style) |
| `mfcc_cnn` | **MFCC sequence** (40 coefficients × 151 frames) | 1D CNN over time, coefficients as channels |
| `mel_cnn` | **log-mel spectrogram** (64 mel bands × 151 frames) | 2D VGG-style CNN, spectrogram treated as an image |
| `crnn` | log-mel spectrogram | CNN → BiGRU → attention pooling over time |
| `wavlm` | **self-supervised speech model**: [WavLM base+](https://huggingface.co/microsoft/wavlm-base-plus), frozen | learned weighted sum of its 13 hidden layers (mean + std pooled over time) → MLP head |

The 7 neural models are defined in `models.py`. The SVM (scikit-learn) is in `train.py`.

## Training and evaluation setup

- **Split:** by actor, so no test voice is heard during training. Train = actors 1-18 (1,080 clips),
  validation = actors 19-20 (120), test = actors 21-24 (240).
- **Audio:** resampled to 16 kHz, leading and trailing silence trimmed, then center-cropped or zero-padded to 3 s.
  `notebook_fnn` keeps the notebook's own loading (22.05 kHz, no trimming).
- **Training loop (all neural models):** AdamW (weight decay 1e-2), one-cycle learning-rate schedule, label smoothing
  0.1, batch size 32. The validation macro-F1 is checked every epoch and the best checkpoint is kept for testing.
  `notebook_fnn` keeps the original architecture, features and learning rate (1e-4), so the comparison is about the
  features and architecture rather than optimiser tricks.
- **Augmentation (waveform-input models):** random circular time shift (up to 0.3 s) and Gaussian noise at 15-40 dB SNR.
  `mfcc_cnn`, `mel_cnn` and `crnn` also get SpecAugment (random time and frequency masks), because they compute
  their front-end (`LogMel` / `MFCC`) inside the model.
- **Metrics:** accuracy and **macro-F1**. *Neutral* has half as many clips as each other emotion (RAVDESS has no
  strong-intensity neutral), so macro-F1 is the fairer score. Per-epoch train/val loss, accuracy, macro-F1 and learning
  rate are saved in each `results/<model>_seed<n>.json`.

## Quick start

```bash
git clone <this-repo> && cd speech-emotion-recognition
pip install -r requirements.txt
python prepare_ravdess.py     # downloads RAVDESS (~200 MB), splits it by actor -> data/ravdess/{train,val,test}
python compare.py             # trains all models (3 seeds), writes results/RESULTS.md and every figure
```

`compare.py` is resumable: it skips any run whose `results/<model>_seed<n>.json` already exists.
The first run caches preprocessed audio and features in `cache/`. Delete that folder if you change the audio.
A GPU is used automatically when available (on Colab: *Runtime → Change runtime type → GPU*).

## Usage

```bash
# Train / evaluate one model (writes results/<model>_seed<n>.json + results/checkpoints/)
python train.py --model crnn
python train.py --model mel_cnn --seed 1

# Experiments: use a separate --out folder so the published results are not overwritten
python train.py --model crnn --epochs 100 --out results/exp_long
python train.py --model wavlm --backbone facebook/wav2vec2-base --out results/exp_w2v2   # any HF wav2vec2 / HuBERT / WavLM

# Your own data: a folder containing train/ val/ test/ subfolders of .wav files
python train.py --model mel_cnn --data /path/to/datasets

# Classify one clip (needs a trained checkpoint; top-3 emotions with probabilities)
python predict.py clip.wav --model wavlm
python predict.py clip.wav --model wavlm --backbone facebook/wav2vec2-base --ckpt-dir results/exp_w2v2/checkpoints

# Rebuild the table and figures from results/*.json
python report.py
```

Labels are read from the file name:

- original RAVDESS names, such as `03-01-05-01-02-01-12.wav`: the 3rd field is the emotion
  (01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised)
- any other `<a>-<b>-<emotion>.wav` name: the last field is the emotion (same codes)

## Repository layout

```
data.py              audio loading, labels, handcrafted + notebook features, caching
models.py            the 7 neural architectures + front-ends (LogMel, MFCC, SpecAugment)
train.py             train/evaluate one model (incl. the SVM); logs per-epoch metrics to results/<model>_seed<n>.json
compare.py           runs every model × seed (resumable), then builds the report
report.py            results table + figures: accuracy comparison, training curves, confusion matrices, per-class F1
predict.py           inference on a single .wav file
prepare_ravdess.py   downloads RAVDESS and makes the speaker-independent split
results/             per-run metrics (JSON, incl. training history), RESULTS.md and the figures
```

## Dataset

[RAVDESS](https://zenodo.org/records/1188976), speech subset: 1,440 clips from 24 professional actors and 8 emotions.
It is licensed CC BY-NC-SA 4.0. `prepare_ravdess.py` downloads it; the audio is not redistributed in this repo.

> Livingstone SR, Russo FA (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).*
> PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391

This project started from a Colab notebook (*speech_sentiment_analysis with FNN*). Its pipeline is kept as the
`notebook_fnn` baseline, so every other model is measured against it.
