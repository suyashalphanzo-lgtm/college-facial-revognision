import numpy as np

from app import config
from app.recognizer import UNKNOWN, find_best_match


def unit(numbers):
    vector = np.array(numbers, dtype=np.float32)
    return vector / np.linalg.norm(vector)


# Real embeddings have 128 numbers, but the matching works the same way with 3.
DATABASE = {"Asha": unit([1, 0, 0]), "Ravi": unit([0, 1, 0])}


def test_picks_the_most_similar_person():
    name, similarity = find_best_match(unit([0.9, 0.1, 0]), DATABASE)
    assert name == "Asha"
    assert similarity > 0.9


def test_unknown_when_nobody_is_similar_enough():
    name, similarity = find_best_match(unit([0, 0, 1]), DATABASE)
    assert name == UNKNOWN
    assert similarity < config.MATCH_THRESHOLD


def test_named_only_when_similarity_reaches_the_threshold():
    def face_with_similarity_to_asha(similarity):
        return unit([similarity, 0, (1 - similarity**2) ** 0.5])

    assert find_best_match(face_with_similarity_to_asha(config.MATCH_THRESHOLD + 0.01), DATABASE)[0] == "Asha"
    assert find_best_match(face_with_similarity_to_asha(config.MATCH_THRESHOLD - 0.01), DATABASE)[0] == UNKNOWN


def test_unknown_when_nobody_is_registered():
    name, _ = find_best_match(unit([1, 0, 0]), {})
    assert name == UNKNOWN
