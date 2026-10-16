"""Download RAVDESS (speech, 1440 clips, 24 actors) and split it BY ACTOR into train/val/test.

Speaker-independent split: no test actor is ever heard during training, so test accuracy
measures generalisation to new voices (a random split leaks speakers and inflates accuracy).

    python prepare_ravdess.py            # -> data/ravdess/{train,val,test}/*.wav
"""
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
SPLITS = {"train": range(1, 19), "val": range(19, 21), "test": range(21, 25)}

raw = Path("data/raw")
zip_path = raw / "ravdess_speech.zip"
if not zip_path.exists():
    raw.mkdir(parents=True, exist_ok=True)
    print("downloading RAVDESS (~200 MB)...")
    urllib.request.urlretrieve(URL, zip_path)
if not (raw / "ravdess").exists():
    zipfile.ZipFile(zip_path).extractall(raw / "ravdess")

out = Path("data/ravdess")
for split, actors in SPLITS.items():
    (out / split).mkdir(parents=True, exist_ok=True)
    for actor in actors:
        for wav in (raw / "ravdess" / f"Actor_{actor:02d}").glob("*.wav"):
            shutil.copy(wav, out / split / wav.name)
    print(f"{split}: {len(list((out / split).glob('*.wav')))} clips (actors {actors.start}-{actors.stop - 1})")
