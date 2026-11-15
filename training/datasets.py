"""Getting the face photos ready for training and testing.  Run:
    python -m training.datasets                        (packs datasets/webface_112x112.zip)
    python -m training.datasets --webface <folder>     (packs a folder of person folders, as on Kaggle)

TRAINING: CASIA-WebFace, already aligned to 112x112 (datasets/webface_112x112.zip, from Kaggle
    "yakhyokhuja/webface-112x112"): 490,623 web photos of 10,572 celebrities, one folder per person.
    pack_webface() copies every photo's bytes into one big file (webface_images.bin) with an index
    (webface_index.npz: where each photo starts, how long it is, and its person number).
    Reading from one big file is much faster than opening 490,623 small ones.

TESTING: LFW (datasets/lfw.tgz and pairs.txt, the figshare copies scikit-learn uses; downloaded here
    if missing). align_lfw() runs the app's own detection (YuNet) and alignment (align_face in
    app/face_engine.py) on every LFW photo, so the tests see faces the way the app does. Saved as
    lfw_aligned.npz.

The training photos were aligned with the same 5-point template as the app's align_face (checked: see
"Our own SFace" in CLAUDE.md), so training faces look like the faces the app will show the model.
"""

import argparse
import hashlib
import os
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from app.config import PROJECT_DIR

# On Kaggle this points to /tmp, which is not saved with the notebook's output (see the notebook).
DATASETS_DIR = Path(os.environ.get("FACE_DATASETS_DIR", PROJECT_DIR / "datasets"))
WEBFACE_ZIP = DATASETS_DIR / "webface_112x112.zip"
WEBFACE_IMAGES = DATASETS_DIR / "webface_images.bin"
WEBFACE_INDEX = DATASETS_DIR / "webface_index.npz"
LFW_TGZ = DATASETS_DIR / "lfw.tgz"
LFW_PAIRS = DATASETS_DIR / "pairs.txt"
LFW_ALIGNED = DATASETS_DIR / "lfw_aligned.npz"

# The figshare copies of LFW and their SHA-256 checksums, as published by scikit-learn.
LFW_TGZ_URL = "https://ndownloader.figshare.com/files/5976018"
LFW_PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"
LFW_TGZ_SHA256 = "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"
LFW_PAIRS_SHA256 = "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- training set


def webface_photos(source: Path):
    """Yields (person folder name, photo bytes) for every photo, from the zip or from a folder."""
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                if entry.filename.lower().endswith(".jpg"):
                    yield Path(entry.filename).parent.name, archive.read(entry)
        return
    # On network storage like Kaggle's dataset folders, every file takes a while to arrive, so
    # reading them one by one managed fewer than 100 photos a second. 64 threads each wait for
    # their own file at the same time. pool.map hands the results back in the original order.
    with ThreadPoolExecutor(64) as pool:
        person_dirs = sorted(source.iterdir())
        photos = [photo for listing in pool.map(lambda d: sorted(d.glob("*.jpg")), person_dirs) for photo in listing]
        print(f"  found {len(photos):,} photos in {len(person_dirs):,} folders", flush=True)
        for start in range(0, len(photos), 10000):  # 10,000 at a time, so memory use stays small
            chunk = photos[start : start + 10000]
            for photo, data in zip(chunk, pool.map(Path.read_bytes, chunk)):
                yield photo.parent.name, data


def pack_webface(source: Path = WEBFACE_ZIP) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    offsets, lengths, folders = [], [], []
    position = 0
    started = time.time()
    with open(WEBFACE_IMAGES, "wb") as images_file:
        for number, (folder, data) in enumerate(webface_photos(source)):
            images_file.write(data)
            offsets.append(position)
            lengths.append(len(data))
            folders.append(folder)
            position += len(data)
            if number % 50000 == 0:
                print(f"  packed {number:,} photos ({time.time() - started:.0f} s)", flush=True)

    # Folders are named id_0 ... id_10571; in that order they become person numbers 0 ... 10571 (the labels).
    names = sorted(set(folders), key=lambda name: int(name.split("_")[1]))
    label_of = {name: label for label, name in enumerate(names)}
    labels = np.array([label_of[folder] for folder in folders], np.int32)
    np.savez(WEBFACE_INDEX, offsets=np.array(offsets, np.int64), lengths=np.array(lengths, np.int32), labels=labels)
    print(f"Training set: {len(labels):,} photos of {len(names):,} people, {position / 1e9:.2f} GB")


def check_webface_photos(count: int = 2000) -> None:
    """Check that the packed photos decode and are all 112x112."""
    index = np.load(WEBFACE_INDEX)
    images = np.memmap(WEBFACE_IMAGES, dtype=np.uint8, mode="r")
    sizes = set()
    count = min(count, len(index["labels"]))
    for number in np.random.default_rng(0).choice(len(index["labels"]), count, replace=False):
        start = index["offsets"][number]
        image = cv2.imdecode(np.asarray(images[start : start + index["lengths"][number]]), cv2.IMREAD_COLOR)
        sizes.add(None if image is None else image.shape)
    print(f"Sizes seen in {count} random training photos: {sizes}")


def webcam_style(image: np.ndarray) -> np.ndarray:
    """Randomly makes a clean training photo look more like a webcam picture ("augmentation").

    The training photos are mostly sharp, well-lit celebrity photos; webcams give small, blurry,
    dark or washed-out faces. Each change happens to only some photos, so most stay clean:
      - 20%: low resolution: shrunk to 24-80 pixels wide and enlarged back (a face far from the camera)
      - 20%: blur (out of focus, or moving)
      - 30%: brightness and contrast changed (dark rooms, bright windows)
    """
    if np.random.rand() < 0.2:
        size = np.random.randint(24, 81)
        small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (112, 112), interpolation=cv2.INTER_LINEAR)
    if np.random.rand() < 0.2:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=np.random.uniform(0.5, 1.5))
    if np.random.rand() < 0.3:
        contrast, brightness = np.random.uniform(0.6, 1.4), np.random.uniform(-40, 40)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)  # also clips to 0..255
    return image


class WebFaceDataset:
    """The training photos for PyTorch: returns (face, person number).

    face: uint8 tensor (3, 112, 112), RGB, flipped left-right half the time ("augmentation":
    a mirrored face is still the same person, so this doubles the variety for free), and with
    random webcam-style changes (see webcam_style).
    """

    def __init__(self, flip: bool = True, webcam_changes: bool = True):
        index = np.load(WEBFACE_INDEX)
        self.offsets, self.lengths, self.labels = index["offsets"], index["lengths"], index["labels"]
        self.num_people = int(self.labels.max()) + 1
        self.flip = flip
        self.webcam_changes = webcam_changes
        self.images = None  # opened in each worker process, see __getitem__

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, number):
        import torch

        if self.images is None:
            # A memory map lets every worker read the big file without loading all of it into memory.
            self.images = np.memmap(WEBFACE_IMAGES, dtype=np.uint8, mode="r")
        start = self.offsets[number]
        image = cv2.imdecode(np.asarray(self.images[start : start + self.lengths[number]]), cv2.IMREAD_COLOR)
        if self.webcam_changes:
            image = webcam_style(image)
        image = image[:, :, ::-1]  # OpenCV gives BGR; the model is trained on RGB, like OpenCV's SFace
        if self.flip and np.random.rand() < 0.5:
            image = image[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))), int(self.labels[number])


# ---------------------------------------------------------------- test set


def align_lfw() -> None:
    from app.face_engine import FaceEngine, align_face

    for path, url, expected in ((LFW_TGZ, LFW_TGZ_URL, LFW_TGZ_SHA256), (LFW_PAIRS, LFW_PAIRS_URL, LFW_PAIRS_SHA256)):
        if not path.exists():
            print(f"Downloading {path.name}...", flush=True)
            DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, path.with_suffix(".download"))
            path.with_suffix(".download").replace(path)
        if sha256(path) != expected:
            raise SystemExit(f"{path.name} does not match its published checksum; download it again.")
    print("LFW files match their published checksums.")

    engine = FaceEngine(load_recognizer=False)  # only detection: our model doesn't exist before training
    # LFW photos are small (250x250) news photos; accept slightly less certain detections than the app.
    engine.detector.setScoreThreshold(0.5)

    names, stems, crops, found_faces, missed = [], [], [], [], {}
    with tarfile.open(LFW_TGZ) as archive:
        for member in archive:
            if not member.name.lower().endswith(".jpg"):
                continue
            data = np.frombuffer(archive.extractfile(member).read(), np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            names.append(Path(member.name).parent.name)
            stems.append(Path(member.name).stem)
            found = engine.detect_faces(image)
            if found:
                # LFW photos are centred on the named person, so take the face closest to the centre.
                centre = np.array(image.shape[1::-1]) / 2
                face = min(found, key=lambda f: np.linalg.norm(f[:2] + f[2:4] / 2 - centre))
                found_faces.append(face)
                crops.append(align_face(image, face))
            else:
                missed[len(crops)] = image
                crops.append(None)
            if len(crops) % 2000 == 0:
                print(f"  {len(crops):,} LFW photos done")

    # Where no face was found, use the average landmark positions of all the other photos instead.
    average_face = np.mean(found_faces, axis=0)
    for number, image in missed.items():
        crops[number] = align_face(image, average_face)

    detected = np.ones(len(crops), bool)
    detected[list(missed)] = False
    np.savez(
        LFW_ALIGNED,
        crops=np.stack(crops),  # (13233, 112, 112, 3), BGR like everything in OpenCV
        names=np.array(names),
        stems=np.array(stems),
        detected=detected,
    )
    print(f"LFW: {len(crops):,} photos aligned; no face found in {len(missed)} (average landmarks used for those).")


def load_lfw_pairs(stems: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The official 6,000 LFW test pairs, as photo numbers.

    Returns (first, second, same): the two photos of each pair and whether they show the same person.
    pairs.txt holds 10 groups ("folds") of 300 same-person pairs followed by 300 different-person pairs.
    """
    number_of = {stem: number for number, stem in enumerate(stems)}
    first, second, same = [], [], []
    for line in LFW_PAIRS.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 3:
            name1, n1, name2, n2 = parts[0], parts[1], parts[0], parts[2]
        else:
            name1, n1, name2, n2 = parts
        first.append(number_of[f"{name1}_{int(n1):04d}"])
        second.append(number_of[f"{name2}_{int(n2):04d}"])
        same.append(len(parts) == 3)
    return np.array(first), np.array(second), np.array(same)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--webface", type=Path, default=WEBFACE_ZIP, help="the zip, or a folder of person folders")
    args = parser.parse_args()
    if not WEBFACE_INDEX.exists():
        pack_webface(args.webface)
    check_webface_photos()
    if not LFW_ALIGNED.exists():
        align_lfw()
