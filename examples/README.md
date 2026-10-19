# Examples

Short scripts that use the repo's modules from Python, in the order you would use them. Run them from the repo root
on a CPU; outputs go to `examples/output/` (gitignored). Scripts 02-05 need the data: `python prepare_ravdess.py`.

| script | what it shows | time (CPU) |
|---|---|---|
| [`01_listen_to_samples.py`](01_listen_to_samples.py) | the 8 sample clips: duration, loudness, pitch and log-mel spectrograms | seconds |
| [`02_train_svm.py`](02_train_svm.py) | quickest real result: SVM on handcrafted features, per-emotion report on unseen actors | seconds (+2 min first-run preprocessing) |
| [`03_overfitting_fix.py`](03_overfitting_fix.py) | same MLP with the original vs the regularised recipe: train/val gap and test accuracy | 2-3 min |
| [`04_speaker_normalisation.py`](04_speaker_normalisation.py) | per-speaker feature normalisation removes voice identity: about +12 points for the SVM | seconds |
| [`05_classify_samples.py`](05_classify_samples.py) | classify the sample clips with a trained checkpoint, true vs predicted emotion | seconds |
| [`06_render_sample_videos.py`](06_render_sample_videos.py) | render each sample as a video with sound into `samples/videos/` (needs ffmpeg) | ~1 min |
