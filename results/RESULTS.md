# Results (RAVDESS, speaker-independent test set: actors 21-24)

| model | input | test accuracy | test macro-F1 | params | train time / run | seeds |
|---|---|---|---|---|---|---|
| `wavlm` | pretrained WavLM states | **77.6** ± 1.3 % | 76.7 ± 1.2 % | 0.40 M | 52 s | 3 |
| `crnn` | log-mel spectrogram | **62.8** ± 0.4 % | 60.8 ± 1.1 % | 0.32 M | 426 s | 3 |
| `mel_cnn` | log-mel spectrogram | **57.1** ± 1.5 % | 48.9 ± 3.4 % | 0.36 M | 445 s | 3 |
| `mfcc_cnn` | MFCC sequence | **55.6** ± 0.5 % | 54.5 ± 0.7 % | 0.17 M | 118 s | 3 |
| `svm` | handcrafted features | **49.2** ± 0.0 % | 48.0 ± 0.0 % | – | 1 s | 3 |
| `mlp` | handcrafted features | **47.4** ± 0.9 % | 46.4 ± 1.0 % | 0.27 M | 212 s | 3 |
| `wave_cnn` | raw waveform | **39.4** ± 2.2 % | 36.6 ± 2.8 % | 0.15 M | 137 s | 3 |
| `notebook_fnn` | notebook features (199-d) | **27.8** ± 3.4 % | 24.0 ± 3.1 % | 1.48 M | 77 s | 3 |
