"""Testing our face recognizer on LFW.  Run:
    python -m training.evaluate                     (our own SFace)
    python -m training.evaluate --compare-opencv    (also OpenCV's pretrained SFace, as a yardstick)

Both tests use the LFW faces aligned by the app's own code (training/datasets.py), and our model is
run from its exported ONNX file by OpenCV's deep-learning engine, exactly as the app runs it.
OpenCV's SFace is not part of the app: --compare-opencv downloads it only to measure how close
our model comes to it, tested in exactly the same way.

1. The standard LFW test ("verification"): 6,000 pairs of photos, half of them the same person.
   A pair counts as "same person" when the two embeddings' similarity reaches a threshold. The
   threshold is chosen on 9 of the 10 groups of pairs and tested on the 10th, ten times over, and
   the accuracy is averaged. Research papers report this number, so it compares us with them.

2. The college test ("identification"), as in "Accuracy measured on the LFW dataset" in CLAUDE.md:
   register 100 people with 5 photos each, then check, at each threshold, how often up to 5 other
   photos of each of them get the right name and how often 1,000 strangers get a name. This is what
   the app actually does, and it is how MATCH_THRESHOLD is chosen.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from app import config
from app.face_engine import download_model, load_network, run_network
from training.datasets import DATASETS_DIR, LFW_ALIGNED, load_lfw_pairs

# The threshold rule: the lowest threshold at which no more than 0.4% of strangers get a name,
# with 100 registered people.
STRANGERS_ALLOWED = 0.004
# OpenCV's pretrained SFace, only for --compare-opencv: the yardstick our model was measured against.
OPENCV_SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
OPENCV_SFACE_FILE = config.MODELS_DIR / "face_recognition_sface_2021dec.onnx"


def load_lfw():
    data = np.load(LFW_ALIGNED)
    return data["crops"], data["names"], data["stems"]


def normalize(embeddings: np.ndarray) -> np.ndarray:
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def embed_with_opencv(model_file: Path, crops: np.ndarray) -> np.ndarray:
    """Embeddings exactly as the app makes them (run_network on each aligned face)."""
    network = load_network(model_file)
    return normalize(np.stack([run_network(network, crop) for crop in crops]))


def embed_with_torch(model, crops: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """The same embeddings straight from PyTorch, much faster on the GPU (used during training)."""
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            rgb = np.ascontiguousarray(crops[start : start + batch_size, :, :, ::-1].transpose(0, 3, 1, 2))
            parts.append(model(torch.from_numpy(rgb).to(device).float()).float().cpu().numpy())
    model.train(was_training)
    return normalize(np.concatenate(parts))


def lfw_verification(embeddings: np.ndarray, first, second, same) -> tuple[float, float, float]:
    """Returns (mean accuracy, its spread over the 10 folds, mean chosen threshold)."""
    similarities = np.sum(embeddings[first] * embeddings[second], axis=1)
    folds = np.arange(len(similarities)) // 600
    thresholds = np.arange(-1.0, 1.0, 0.001)
    accuracies, chosen = [], []
    for fold in range(10):
        tune = folds != fold
        tune_accuracy = ((similarities[tune][None, :] >= thresholds[:, None]) == same[tune][None, :]).mean(axis=1)
        threshold = thresholds[tune_accuracy.argmax()]
        accuracies.append(((similarities[~tune] >= threshold) == same[~tune]).mean())
        chosen.append(threshold)
    return float(np.mean(accuracies)), float(np.std(accuracies)), float(np.mean(chosen))


def identification_test(
    embeddings, names, thresholds, photos_per_person=5, registered_count=100, stranger_count=1000, draws=(1, 2, 3)
):
    """Returns, for each threshold, (share of registered people's photos named correctly,
    share of strangers given a name), averaged over several random draws of people."""
    photos_of = defaultdict(list)
    for number, name in enumerate(names):
        photos_of[name].append(number)
    candidates = sorted(name for name, photos in photos_of.items() if len(photos) >= 6)

    correct = np.zeros(len(thresholds))
    named = np.zeros(len(thresholds))
    for seed in draws:
        rng = np.random.default_rng(seed)
        registered = rng.choice(candidates, registered_count, replace=False)
        database, tests, test_labels = [], [], []
        for label, name in enumerate(registered):
            photos = rng.permutation(photos_of[name])
            average = embeddings[photos[:photos_per_person]].mean(axis=0)
            database.append(average / np.linalg.norm(average))
            tests += list(photos[5:10])  # the same test photos whatever photos_per_person is
            test_labels += [label] * len(photos[5:10])
        others = sorted(set(photos_of) - set(registered))
        strangers = [rng.choice(photos_of[name]) for name in rng.choice(others, stranger_count, replace=False)]

        database = np.array(database)
        test_similarities = embeddings[tests] @ database.T
        best_person, best_similarity = test_similarities.argmax(axis=1), test_similarities.max(axis=1)
        stranger_best = (embeddings[strangers] @ database.T).max(axis=1)
        for number, threshold in enumerate(thresholds):
            correct[number] += np.mean((best_person == np.array(test_labels)) & (best_similarity >= threshold))
            named[number] += np.mean(stranger_best >= threshold)
    return correct / len(draws), named / len(draws)


def choose_threshold(embeddings, names) -> float:
    """The lowest threshold (to 0.01) at which no more than 0.4% of strangers get a name."""
    thresholds = np.round(np.arange(0.20, 0.80, 0.01), 2)
    _, named = identification_test(embeddings, names, thresholds)
    return float(thresholds[np.argmax(named <= STRANGERS_ALLOWED)])


def report(title: str, embeddings: np.ndarray, names, stems) -> None:
    accuracy, spread, threshold = lfw_verification(embeddings, *load_lfw_pairs(stems))
    print(f"\n===== {title}")
    print(f"Standard LFW test (6,000 pairs): {accuracy:.2%} correct (+/- {spread:.2%}), best threshold about {threshold:.2f}")

    chosen = choose_threshold(embeddings, names)
    print(f"Lowest threshold giving at most {STRANGERS_ALLOWED:.1%} of strangers a name: {chosen:.2f}")
    print("College test, 100 registered people:")
    print("| Photos per person | Threshold | Registered people named correctly | Strangers given a name |")
    print("| --- | --- | --- | --- |")
    for photos_per_person in (5, 3, 1):
        thresholds = sorted({0.363, 0.40, 0.46, 0.50, chosen}) if photos_per_person == 5 else [chosen]
        correct, named = identification_test(embeddings, names, thresholds, photos_per_person)
        for threshold, c, n in zip(thresholds, correct, named):
            print(f"| {photos_per_person} | {threshold:.3g} | {c:.1%} | {n:.1%} |")


def cached_opencv_embeddings(model_file: Path, crops: np.ndarray, mirror: bool = False) -> np.ndarray:
    """LFW embeddings as the app makes them, saved so the next run is quick. With mirror=True,
    each face's embedding is averaged with its mirror image's, like the app's mirror trick."""
    parts = [("", crops)]
    if mirror:
        parts.append(("_mirrored", np.ascontiguousarray(crops[:, :, ::-1])))  # left-right flip
    total = 0
    for suffix, faces in parts:
        cache = DATASETS_DIR / f"lfw_embeddings_{model_file.stem}{suffix}.npy"
        if cache.exists() and cache.stat().st_mtime > model_file.stat().st_mtime:
            embeddings = np.load(cache)
        else:
            print(f"Making LFW{suffix.replace('_', ' ')} embeddings with {model_file.name} (about a minute)...")
            embeddings = embed_with_opencv(model_file, faces)
            np.save(cache, embeddings)
        total = total + embeddings
    return normalize(total)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", choices=["app", "on", "off"], default="app",
                        help="the mirror trick: on for our model as in app/config.py, off for OpenCV's (default), or on/off for both")
    parser.add_argument("--compare-opencv", action="store_true",
                        help="also test OpenCV's pretrained SFace the same way (downloaded for this, 37 MB)")
    args = parser.parse_args()
    crops, names, stems = load_lfw()
    models = [("Our own SFace", config.OUR_MODEL_FILE, config.MIRROR_TRICK)]
    if args.compare_opencv:
        download_model(OPENCV_SFACE_URL, OPENCV_SFACE_FILE)
        models.insert(0, ("OpenCV's SFace, for comparison", OPENCV_SFACE_FILE, False))
    for title, model_file, mirror_as_in_app in models:
        if not model_file.exists():
            print(f"\n{model_file.name} does not exist yet: train and export our model first.")
            continue
        mirror = mirror_as_in_app if args.mirror == "app" else args.mirror == "on"
        title += ", with the mirror trick" if mirror else ""
        report(title, cached_opencv_embeddings(model_file, crops, mirror), names, stems)
