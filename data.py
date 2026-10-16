"""Loading audio, labels and the cached arrays every model trains from.

Expected layout (same as the original notebook):
    <data_dir>/train/*.wav
    <data_dir>/val/*.wav
    <data_dir>/test/*.wav

Labels come from the file name:
    - original RAVDESS names, 7 fields "03-01-05-01-02-01-12.wav" -> 3rd field is the emotion
    - notebook-style names "A0-22-05.wav"                         -> last field is the emotion
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
SR = 16_000
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(SR * CLIP_SECONDS)


def label_from_path(path):
    parts = Path(path).stem.split("-")
    field = parts[2] if len(parts) == 7 else parts[-1]
    return int(field) - 1


def load_trimmed(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)  # drop the leading/trailing silence
    return y


def fix_length(y, n=CLIP_SAMPLES):
    """Center-crop or zero-pad to exactly n samples."""
    if len(y) >= n:
        start = (len(y) - n) // 2
        return y[start:start + n]
    pad = n - len(y)
    return np.pad(y, (pad // 2, pad - pad // 2))


def handcrafted_features(y):
    """The notebook's feature set (MFCC, chroma, mel, contrast, tonnetz), pooled with mean AND std."""
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    feats = [
        librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40),
        librosa.feature.chroma_stft(S=stft**2, sr=SR),
        librosa.power_to_db(librosa.feature.melspectrogram(S=stft**2, sr=SR, n_mels=64)),
        librosa.feature.spectral_contrast(S=stft, sr=SR),
        librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=SR),
    ]
    return np.concatenate([np.r_[f.mean(axis=1), f.std(axis=1)] for f in feats]).astype(np.float32)


def notebook_features(path):
    """Exact recipe of the original Colab notebook: 199 dims, 22.05 kHz, no trimming, per-clip min-max."""
    x, sr = librosa.load(path)
    stft = np.abs(librosa.stft(x))
    fts = np.hstack([
        np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=47).T, axis=0)[1:],
        np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0),
        np.mean(librosa.feature.melspectrogram(y=x, sr=sr).T, axis=0),
        np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0),
        np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sr).T, axis=0),
    ])
    return ((fts - fts.min()) / (fts.max() - fts.min())).astype(np.float32)


def _process(path):
    y = load_trimmed(path)
    return fix_length(y).astype(np.float32), handcrafted_features(y), notebook_features(path), label_from_path(path)


def load_split(data_dir, split, cache_dir="cache"):
    """Returns dict(wave=[N, CLIP_SAMPLES], feats=[N, D], nb_feats=[N, 199], y=[N]); computed once, then cached."""
    cache = Path(cache_dir) / f"{Path(data_dir).name}_{split}.npz"
    if cache.exists():
        return dict(np.load(cache))
    paths = sorted(str(p) for p in (Path(data_dir) / split).glob("*.wav"))
    assert paths, f"no .wav files in {Path(data_dir) / split}"
    with ProcessPoolExecutor() as ex:
        out = list(tqdm(ex.map(_process, paths, chunksize=8), total=len(paths), desc=f"preprocess {split}"))
    wave, feats, nb_feats, y = map(np.stack, zip(*out))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, wave=wave, feats=feats, nb_feats=nb_feats, y=y.astype(np.int64))
    return dict(np.load(cache))
