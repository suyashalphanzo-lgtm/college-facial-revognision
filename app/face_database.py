"""Step 3: remember the people we know (registration).

Every folder inside data/people/ is one person, and the folder name is their name:

    data/people/Aaditya/photo1.jpg
    data/people/Aaditya/photo2.jpg
    data/people/Priya/selfie.png

The embeddings of all of a person's photos are averaged into one embedding for that person.
Everyone is saved to data/face_database_ours.json as {"name": [128 numbers], ...}: the numbers
our own SFace gives them. Numbers from a different model can't be compared with these, so after
the model changes, delete the file and everyone is registered again from the photo folders.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from app import config
from app.face_engine import FaceEngine

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_database(
    engine: FaceEngine,
    people_dir: Path = config.PEOPLE_DIR,
    database_file: Path = config.DATABASE_FILE,
) -> dict[str, np.ndarray]:
    """Read every person's photos and save one embedding per person."""
    people_dir.mkdir(parents=True, exist_ok=True)
    database = {}
    for person_dir in sorted(path for path in people_dir.iterdir() if path.is_dir()):
        embedding = embed_person(engine, person_dir)
        if embedding is not None:
            database[person_dir.name] = embedding

    save_database(database, database_file)
    print(f"People registered: {len(database)}")
    return database


def count_photos(person_dir: Path) -> int:
    """How many photos are in one person's folder."""
    if not person_dir.is_dir():
        return 0
    return sum(1 for path in person_dir.iterdir() if path.suffix.lower() in PHOTO_EXTENSIONS)


def embed_person(engine: FaceEngine, person_dir: Path) -> np.ndarray | None:
    """Average embedding of all the photos in one person's folder (None if no photo is usable)."""
    embeddings = []
    for photo in sorted(person_dir.iterdir()):
        if photo.suffix.lower() not in PHOTO_EXTENSIONS:
            continue
        image = cv2.imread(str(photo))
        if image is None:
            print(f"  Skipped {person_dir.name}/{photo.name}: could not read the file")
            continue
        faces = engine.detect_faces(image)
        if not faces:
            print(f"  Skipped {person_dir.name}/{photo.name}: no face found")
            continue
        # If several faces are visible, the biggest one is assumed to be the person.
        biggest_face = max(faces, key=lambda face: face[2] * face[3])
        embeddings.append(engine.get_embedding(image, biggest_face))

    if not embeddings:
        print(f"{person_dir.name}: no usable photos, not registered")
        return None
    print(f"{person_dir.name}: {len(embeddings)} photo(s) used")
    if len(embeddings) < 3:
        print(f"  Tip: add more photos of {person_dir.name} (5 is best) so they are recognized more reliably")
    average = np.mean(embeddings, axis=0)
    return average / np.linalg.norm(average)


def save_database(database: dict[str, np.ndarray], database_file: Path = config.DATABASE_FILE) -> None:
    database_file.parent.mkdir(parents=True, exist_ok=True)
    data = {name: [round(float(number), 6) for number in embedding] for name, embedding in database.items()}
    database_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_database(database_file: Path = config.DATABASE_FILE) -> dict[str, np.ndarray]:
    if not database_file.exists():
        return {}
    data = json.loads(database_file.read_text(encoding="utf-8"))
    return {name: np.array(numbers, dtype=np.float32) for name, numbers in data.items()}


def load_or_build_database(
    engine: FaceEngine,
    people_dir: Path = config.PEOPLE_DIR,
    database_file: Path = config.DATABASE_FILE,
) -> dict[str, np.ndarray]:
    """The saved database. If there is none yet but there are people folders (for example after
    the file was deleted), everyone is registered first."""
    has_people = people_dir.is_dir() and any(path.is_dir() for path in people_dir.iterdir())
    if database_file.exists() or not has_people:
        return load_database(database_file)
    print(f"No {database_file.name} yet: registering everyone in {people_dir.name}/ with this model...")
    return build_database(engine, people_dir, database_file)
