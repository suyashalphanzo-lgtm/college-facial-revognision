import cv2
import numpy as np

from app.face_database import load_database, load_or_build_database, save_database


class FakeEngine:
    """Stands in for the face models: finds one face in every photo and gives it the same embedding."""

    def detect_faces(self, image):
        return [np.array([0, 0, 10, 10] + [0] * 10 + [0.9], dtype=np.float32)]

    def get_embedding(self, image, face):
        return np.array([0.6, 0.8], dtype=np.float32)


def test_save_then_load_gives_back_the_same_embeddings(tmp_path):
    database_file = tmp_path / "face_database.json"
    database = {"Asha": np.array([0.6, 0.8], dtype=np.float32)}

    save_database(database, database_file)
    loaded = load_database(database_file)

    assert list(loaded) == ["Asha"]
    assert np.allclose(loaded["Asha"], database["Asha"], atol=1e-6)


def test_missing_database_file_means_nobody_is_registered(tmp_path):
    assert load_database(tmp_path / "missing.json") == {}


def test_first_start_with_a_recognizer_registers_everyone(tmp_path):
    people_dir = tmp_path / "people"
    (people_dir / "Asha").mkdir(parents=True)
    cv2.imwrite(str(people_dir / "Asha" / "photo.png"), np.zeros((20, 20, 3), np.uint8))
    database_file = tmp_path / "face_database_ours.json"

    database = load_or_build_database(FakeEngine(), people_dir, database_file)

    assert list(database) == ["Asha"]
    assert database_file.exists()


def test_an_existing_database_is_loaded_not_rebuilt(tmp_path):
    people_dir = tmp_path / "people"
    (people_dir / "Asha").mkdir(parents=True)
    database_file = tmp_path / "face_database.json"
    save_database({"Priya": np.array([1.0, 0.0], dtype=np.float32)}, database_file)

    database = load_or_build_database(None, people_dir, database_file)  # None: the models must not be used

    assert list(database) == ["Priya"]


def test_nothing_is_built_without_people_folders(tmp_path):
    database_file = tmp_path / "face_database.json"

    assert load_or_build_database(None, tmp_path / "people", database_file) == {}
    assert not database_file.exists()
