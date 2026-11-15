"""Menu option 3: find and name every face in a photo file."""

from pathlib import Path

import cv2
import numpy as np

from app import config
from app.face_database import load_database
from app.face_engine import FaceEngine
from app.recognizer import Recognition, draw_results, recognize_faces


def recognize_photo(engine: FaceEngine, photo_path: Path) -> None:
    database = load_database()
    if not database:
        print("No one is registered yet. Add people first (menu option 1 or 2).")
        return
    if not photo_path.is_file():
        print(f"There is no file at: {photo_path}")
        return
    image = cv2.imread(str(photo_path))
    if image is None:
        print(f"Could not open the photo: {photo_path}")
        return

    results, result_path = label_and_save(engine, database, image, photo_path)
    print(f"Found {len(results)} face(s):")
    for result in results:
        print(f"  {result.name} (similarity {result.similarity:.2f})")
    print(f"Saved the labelled photo to {result_path}")
    show_image(image, "Result (press any key to close)")


def label_and_save(
    engine: FaceEngine, database: dict[str, np.ndarray], image: np.ndarray, photo_path: Path
) -> tuple[list[Recognition], Path]:
    """Name every face, draw the names onto the image, and save a copy in data/results/."""
    results = recognize_faces(engine, database, image)
    draw_results(image, results)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = config.RESULTS_DIR / f"{photo_path.stem}_result.jpg"
    cv2.imwrite(str(result_path), image)
    return results, result_path


def show_image(image: np.ndarray, title: str) -> None:
    """Show an image, shrunk to fit the screen, until a key is pressed or the window is closed."""
    scale = min(1.0, 1200 / image.shape[1], 700 / image.shape[0])
    cv2.imshow(title, cv2.resize(image, None, fx=scale, fy=scale))
    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(100) != -1:
            break
    cv2.destroyAllWindows()
