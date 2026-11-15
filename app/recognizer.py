"""Step 4: decide WHO each face is by comparing its embedding with every registered person.

Two embeddings are compared with "cosine similarity". Because every embedding has length 1,
that is simply the dot product of the two lists of numbers:
    close to 1.0  ->  the same face
    close to 0.0  ->  unrelated faces
A face only gets a name if its best similarity reaches config.MATCH_THRESHOLD.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from app import config
from app.face_engine import FaceEngine

UNKNOWN = "Unknown"


@dataclass
class Recognition:
    box: tuple[int, int, int, int]  # x, y, width, height of the face in the image
    name: str
    similarity: float


def find_best_match(embedding: np.ndarray, database: dict[str, np.ndarray]) -> tuple[str, float]:
    """Return the most similar registered person, or UNKNOWN if nobody is similar enough."""
    best_name, best_similarity = UNKNOWN, -1.0
    for name, known_embedding in database.items():
        similarity = float(np.dot(embedding, known_embedding))
        if similarity > best_similarity:
            best_name, best_similarity = name, similarity

    if best_similarity < config.MATCH_THRESHOLD:
        return UNKNOWN, best_similarity
    return best_name, best_similarity


def recognize_faces(
    engine: FaceEngine, database: dict[str, np.ndarray], image: np.ndarray
) -> list[Recognition]:
    """Steps 1, 2 and 4 together: find every face in the image and put a name to it."""
    results = []
    for face in engine.detect_faces(image):
        embedding = engine.get_embedding(image, face)
        name, similarity = find_best_match(embedding, database)
        x, y, width, height = (int(value) for value in face[:4])
        results.append(Recognition((x, y, width, height), name, similarity))
    return results


def draw_results(image: np.ndarray, results: list[Recognition]) -> None:
    """Draw a box and a "name (similarity)" label on the image for every face."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = max(0.6, image.shape[1] / 1600)  # bigger text on bigger images
    thickness = max(1, round(text_size * 2))
    for result in results:
        x, y, width, height = result.box
        color = (0, 0, 255) if result.name == UNKNOWN else (0, 180, 0)  # OpenCV colors are (blue, green, red)
        label = f"{result.name} ({result.similarity:.2f})"
        (label_width, label_height), _ = cv2.getTextSize(label, font, text_size, thickness)
        label_top = max(0, y - label_height - 10)

        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness + 1)
        cv2.rectangle(image, (x, label_top), (x + label_width + 10, label_top + label_height + 10), color, cv2.FILLED)
        cv2.putText(image, label, (x + 5, label_top + label_height + 5), font, text_size, (255, 255, 255), thickness)
