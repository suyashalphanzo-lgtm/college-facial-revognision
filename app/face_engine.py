"""The two AI models that do the real work, and the alignment step between them.

Step 1, detect_faces():  WHERE are the faces in an image?  (OpenCV's pretrained YuNet model)
Step 2, get_embedding(): turn one face into 128 numbers, called an "embedding".
         First align_face() turns the face upright and cuts it out at 112x112 pixels. Then our own
         SFace, trained with the code in training/, turns it into the numbers.
         Photos of the same person give similar numbers, photos of different people don't.
"""

import urllib.request
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from app import config

# Where alignment puts the 5 face points in the 112x112 face picture: right eye, left eye, nose tip,
# right and left mouth corner. This is the standard "ArcFace" template. Our model was trained on faces
# aligned to it, and OpenCV's own alignment function (alignCrop) uses the same one.
TEMPLATE_POINTS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]]
)
FACE_SIZE = 112


def download_model(url: str, path: Path) -> Path:
    """Download a model file the first time it is needed."""
    if not path.exists():
        print(f"Downloading {path.name} (only needed once)...")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Download under a temporary name so an interrupted download is never mistaken for a model.
        temporary_path = path.with_suffix(".download")
        urllib.request.urlretrieve(url, temporary_path)
        temporary_path.replace(path)
    return path


@contextmanager
def opencv_warnings_hidden():
    """OpenCV 5 prints a harmless "Targets are not supported" warning while it loads these models."""
    previous_log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    try:
        yield
    finally:
        cv2.utils.logging.setLogLevel(previous_log_level)


def similarity_transform(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The turn, scale and shift (no stretching) that move `points` as close as possible onto `target`.

    Returned as the 2x3 matrix cv2.warpAffine expects. It is the least-squares fit known as Umeyama's
    method, the same calculation OpenCV's alignCrop does.
    """
    points_mean, target_mean = points.mean(axis=0), target.mean(axis=0)
    points_centred, target_centred = points - points_mean, target - target_mean
    covariance = target_centred.T @ points_centred / len(points)
    u, spread, vt = np.linalg.svd(covariance)
    # Only a turn, never a mirror image: if the closest fit would mirror the face, it is not allowed to.
    signs = np.array([1.0, 1.0 if np.linalg.det(covariance) >= 0 else -1.0])
    turn = u @ np.diag(signs) @ vt
    scale = spread @ signs / points_centred.var(axis=0).sum()
    shift = target_mean - scale * turn @ points_mean
    return np.hstack([scale * turn, shift[:, None]])


def align_face(image: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Cut one detected face out of the image as a 112x112 picture, turned and scaled so that its
    5 points land as close as possible on the template: eyes level, always the same size and place."""
    points = face[4:14].reshape(5, 2).astype(np.float64)
    matrix = similarity_transform(points, TEMPLATE_POINTS)
    return cv2.warpAffine(image, matrix, (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_LINEAR)


def load_network(model_file: Path) -> cv2.dnn.Net:
    """Load a face network saved as ONNX into OpenCV's deep-learning engine, which runs it."""
    with opencv_warnings_hidden():
        return cv2.dnn.readNetFromONNX(str(model_file))


def run_network(network: cv2.dnn.Net, aligned_face: np.ndarray) -> np.ndarray:
    """The 128 numbers a face network gives one aligned face (not yet scaled to length 1).

    Our network takes RGB pixels from 0 to 255 and scales them itself (see training/mobilefacenet.py),
    so the only preparation is turning OpenCV's BGR colour order into RGB.
    """
    network.setInput(cv2.dnn.blobFromImage(aligned_face, swapRB=True))
    return network.forward().flatten()


class FaceEngine:
    def __init__(self, load_recognizer: bool = True):
        """load_recognizer=False loads only the detector: training/datasets.py uses that to align
        the LFW test photos before our model has been trained."""
        detector_file = download_model(config.DETECTOR_MODEL_URL, config.DETECTOR_MODEL_FILE)
        with opencv_warnings_hidden():
            self.detector = cv2.FaceDetectorYN.create(
                str(detector_file), "", (320, 320), config.DETECTION_CONFIDENCE
            )
        self.network = None
        if load_recognizer:
            if not config.OUR_MODEL_FILE.exists():
                raise SystemExit(
                    f"{config.OUR_MODEL_FILE} is missing. It comes with the project (download the project "
                    "again), or make it from the training results with: python -m training.export_onnx"
                )
            self.network = load_network(config.OUR_MODEL_FILE)

    def detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        """Step 1: find every face in an image.

        Each face is a row of 15 numbers:
            [0:4]   the box around the face: x, y, width, height
            [4:14]  five (x, y) points: right eye, left eye, nose tip, right and left mouth corner
            [14]    how confident the detector is that this is a face (0 to 1)
        """
        height, width = image.shape[:2]
        scale = min(1.0, config.DETECTION_MAX_SIZE / max(height, width))
        small_image = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image

        self.detector.setInputSize((small_image.shape[1], small_image.shape[0]))
        _, faces = self.detector.detect(small_image)
        if faces is None:
            return []

        faces[:, :14] /= scale  # boxes and points back to the size of the original image
        return list(faces)

    def get_embedding(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """Step 2: turn one detected face into an embedding (128 numbers, scaled to length 1)."""
        aligned_face = align_face(image, face)
        embedding = run_network(self.network, aligned_face)
        embedding /= np.linalg.norm(embedding)
        if config.MIRROR_TRICK:
            # The mirror image is the same person seen from a slightly different side; averaging
            # the two evens out how the model sees each half of the face.
            mirrored = run_network(self.network, cv2.flip(aligned_face, 1))
            embedding = embedding + mirrored / np.linalg.norm(mirrored)
        return embedding / np.linalg.norm(embedding)
