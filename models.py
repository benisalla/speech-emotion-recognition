"""Every architecture in the comparison, plus the front-ends that turn a waveform into its input.

| name         | input                          | idea                                               |
|--------------|--------------------------------|----------------------------------------------------|
| notebook_fnn | notebook features (199-d)      | the original Colab FNN, reproduced as the baseline |
| svm          | handcrafted features           | classic ML baseline (lives in train.py, sklearn)   |
| mlp          | handcrafted features           | the notebook's FNN idea, regularised               |
| wave_cnn     | raw waveform                   | 1D CNN that learns its own filterbank (M5-style)   |
| mfcc_cnn     | MFCC sequence                  | 1D CNN over time, MFCCs as channels                |
| mel_cnn      | log-mel spectrogram            | 2D CNN, spectrogram treated as an image            |
| crnn         | log-mel spectrogram            | CNN for local patterns + BiGRU/attention over time |
| wavlm        | pretrained WavLM hidden states | frozen self-supervised model + small head          |
"""
import librosa
import numpy as np
import scipy.fft
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import SR

N_CLASSES = 8


# ---------------------------------------------------------------- front-ends (waveform -> features)

def spec_augment(x, n_masks=2, max_f=8, max_t=12):
    """SpecAugment: zero out random frequency bands and time spans. x: [B, F, T]."""
    B, n_f, n_t = x.shape
    for _ in range(n_masks):
        for axis, size, max_w in ((1, n_f, max_f), (2, n_t, max_t)):
            w = torch.randint(0, max_w + 1, (B, 1, 1), device=x.device)
            start = (torch.rand(B, 1, 1, device=x.device) * (size - w)).long()
            idx = torch.arange(size, device=x.device).view((1, -1, 1) if axis == 1 else (1, 1, -1))
            x = x.masked_fill((idx >= start) & (idx < start + w), 0.0)
    return x


class LogMel(nn.Module):
    """[B, T] waveform -> [B, n_mels, frames] log-mel, per-utterance normalised (20 ms hop, 40 ms window)."""

    def __init__(self, n_mels=64, n_fft=1024, hop=320, win=640, augment=True):
        super().__init__()
        self.n_fft, self.hop, self.win, self.augment = n_fft, hop, win, augment
        self.register_buffer("fb", torch.tensor(librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels, fmin=20)))
        self.register_buffer("window", torch.hann_window(win))

    def log_mel(self, wave):
        spec = torch.stft(wave, self.n_fft, self.hop, self.win, self.window, return_complex=True).abs() ** 2
        return torch.log(self.fb @ spec + 1e-6)

    def forward(self, wave):
        x = self.log_mel(wave)
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        return spec_augment(x) if self.training and self.augment else x


class MFCC(LogMel):
    """[B, T] -> [B, n_mfcc, frames]: DCT of the log-mel, then per-coefficient mean/variance norm."""

    def __init__(self, n_mfcc=40, n_mels=64, **kw):
        super().__init__(n_mels=n_mels, **kw)
        self.register_buffer("dct", torch.tensor(scipy.fft.dct(np.eye(n_mels), norm="ortho", axis=0)[:n_mfcc], dtype=torch.float32))

    def forward(self, wave):
        x = self.dct @ self.log_mel(wave)
        x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-5)
        return spec_augment(x, max_f=4) if self.training and self.augment else x


# ---------------------------------------------------------------- shared pieces

def mean_max_pool(x):
    """Pool the last (time) axis two ways and concatenate: [B, C, T] -> [B, 2C]."""
    return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)


class AttentionPool(nn.Module):
    """Learned weighted average over time: [B, T, D] -> [B, D]."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        return (torch.softmax(self.score(x), dim=1) * x).sum(dim=1)


def head(in_dim, n_classes=N_CLASSES, hidden=256, p=0.3):
    return nn.Sequential(nn.Dropout(p), nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, n_classes))


def conv1d_block(cin, cout, k, stride=1, pool=2):
    return nn.Sequential(nn.Conv1d(cin, cout, k, stride, padding=k // 2, bias=False),
                         nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(pool))


def conv2d_block(cin, cout, pool=(2, 2)):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.MaxPool2d(pool), nn.Dropout(0.1))


# ---------------------------------------------------------------- models

class NotebookFNN(nn.Module):
    """The original notebook's network, unchanged: 6 ReLU dense layers, no normalisation or dropout."""

    def __init__(self, in_dim, sizes=(1000, 750, 500, 250, 100, 50)):
        super().__init__()
        layers, d = [], in_dim
        for h in sizes:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, N_CLASSES))

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    """The notebook's FNN on pooled handcrafted features, with BatchNorm + dropout so it doesn't overfit."""

    def __init__(self, in_dim, hidden=(512, 256), p=0.4):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, N_CLASSES))

    def forward(self, x):
        return self.net(x)


class WaveCNN(nn.Module):
    """Raw waveform in. The first wide, strided conv acts as a learned filterbank."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv1d_block(1, 32, 64, stride=16, pool=4),  # 48000 samples -> 750 steps (~4 ms filters)
            conv1d_block(32, 64, 3, pool=4),             # -> 187
            conv1d_block(64, 128, 3, pool=2),            # -> 93
            conv1d_block(128, 128, 3, pool=1),
        )
        self.head = head(2 * 128)

    def forward(self, wave):
        wave = (wave - wave.mean(1, keepdim=True)) / (wave.std(1, keepdim=True) + 1e-5)
        return self.head(mean_max_pool(self.net(wave.unsqueeze(1))))


class MFCCCNN(nn.Module):
    """MFCC frames in; 1D convolutions slide over time with the 40 coefficients as channels."""

    def __init__(self, n_mfcc=40):
        super().__init__()
        self.frontend = MFCC(n_mfcc=n_mfcc)
        self.net = nn.Sequential(conv1d_block(n_mfcc, 64, 5), conv1d_block(64, 128, 5),
                                 conv1d_block(128, 128, 3, pool=1))
        self.head = head(2 * 128)

    def forward(self, wave):
        return self.head(mean_max_pool(self.net(self.frontend(wave))))


class MelCNN(nn.Module):
    """Log-mel spectrogram in, treated as a 1-channel image (VGG-style blocks)."""

    def __init__(self, channels=(16, 32, 64, 128)):
        super().__init__()
        self.frontend = LogMel()
        blocks, cin = [], 1
        for c in channels:
            blocks.append(conv2d_block(cin, c))
            cin = c
        self.net = nn.Sequential(*blocks)
        self.head = head(2 * cin)

    def forward(self, wave):
        x = self.net(self.frontend(wave).unsqueeze(1))  # [B, C, F', T']
        return self.head(mean_max_pool(x.mean(dim=2)))   # average over frequency, pool over time


class CRNN(nn.Module):
    """Log-mel in. CNN extracts local spectro-temporal patterns, a BiGRU models how they evolve over time."""

    def __init__(self, n_mels=64, rnn_dim=96):
        super().__init__()
        self.frontend = LogMel(n_mels=n_mels)
        self.cnn = nn.Sequential(conv2d_block(1, 16), conv2d_block(16, 32), conv2d_block(32, 64, pool=(2, 1)))
        self.proj = nn.Linear(64 * (n_mels // 8), 128)
        self.rnn = nn.GRU(128, rnn_dim, batch_first=True, bidirectional=True)
        self.pool = AttentionPool(2 * rnn_dim)
        self.head = head(2 * rnn_dim)

    def forward(self, wave):
        x = self.cnn(self.frontend(wave).unsqueeze(1))   # [B, C, F', T']
        x = x.flatten(1, 2).transpose(1, 2)              # [B, T', C*F']
        x, _ = self.rnn(F.relu(self.proj(x)))
        return self.head(self.pool(x))


class SSLHead(nn.Module):
    """Head on a frozen self-supervised model (WavLM / wav2vec2).

    Input is the precomputed per-layer [mean, std] of the hidden states: [B, n_layers, 2*D].
    A softmax-weighted sum over layers lets the model pick which layers carry emotion (SUPERB-style).
    """

    def __init__(self, n_layers, dim):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.head = head(dim, hidden=256, p=0.4)

    def forward(self, x):
        return self.head((torch.softmax(self.layer_logits, 0)[None, :, None] * x).sum(dim=1))


@torch.no_grad()
def ssl_features(waves, backbone="microsoft/wavlm-base-plus", batch_size=8):
    """Run a frozen pretrained speech model once; return per-layer [mean, std] pooled over time."""
    from transformers import AutoModel
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(backbone).eval().to(device)
    out = []
    for i in tqdm(range(0, len(waves), batch_size), desc=f"{backbone} features"):
        x = torch.as_tensor(waves[i:i + batch_size], device=device)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
        hs = torch.stack(model(x, output_hidden_states=True).hidden_states, dim=1)  # [B, L, T, D]
        out.append(torch.cat([hs.mean(2), hs.std(2)], dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# input: which array the model consumes
#   "nb_feats" = notebook features, "feats" = handcrafted, "wave" = raw audio, "ssl" = pretrained states
MODELS = {
    "notebook_fnn": dict(input="nb_feats", build=lambda shape: NotebookFNN(shape[-1]), lr=1e-4, epochs=300, patience=40),
    "mlp":          dict(input="feats", build=lambda shape: MLP(shape[-1]),            lr=1e-3, epochs=300, patience=40),
    "wave_cnn":     dict(input="wave",  build=lambda shape: WaveCNN(),                 lr=3e-3, epochs=40,  patience=15),
    "mfcc_cnn":     dict(input="wave",  build=lambda shape: MFCCCNN(),                 lr=3e-3, epochs=40,  patience=15),
    "mel_cnn":      dict(input="wave",  build=lambda shape: MelCNN(),                  lr=3e-3, epochs=40,  patience=15),
    "crnn":         dict(input="wave",  build=lambda shape: CRNN(),                    lr=2e-3, epochs=40,  patience=15),
    "wavlm":        dict(input="ssl",   build=lambda shape: SSLHead(*shape[-2:]),      lr=1e-3, epochs=200, patience=40),
}
