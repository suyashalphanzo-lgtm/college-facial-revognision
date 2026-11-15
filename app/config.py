"""Every setting of the project in one place."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

PEOPLE_DIR = PROJECT_DIR / "data" / "people"
# The numbers our own SFace gives each registered person.
DATABASE_FILE = PROJECT_DIR / "data" / "face_database_ours.json"
ATTENDANCE_FILE = PROJECT_DIR / "data" / "attendance.csv"
RESULTS_DIR = PROJECT_DIR / "data" / "results"
MODELS_DIR = PROJECT_DIR / "models"

# Step 1, finding faces: OpenCV's pretrained YuNet face detector, from the official OpenCV model zoo.
# It is downloaded automatically on first run.
DETECTOR_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DETECTOR_MODEL_FILE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
# Step 2, recognizing faces: our own SFace, trained with the code in training/ (see "Our own SFace"
# in CLAUDE.md) and saved by "python -m training.export_onnx". It comes with the project.
OUR_MODEL_FILE = MODELS_DIR / "our_sface.onnx"

# How sure (0 to 1) the detector must be before it reports a face.
DETECTION_CONFIDENCE = 0.8
# Photos bigger than this (longest side, in pixels) are shrunk before searching for faces,
# because the detector misses faces that are too big (like close-up selfies).
# Raise it to 1280 only for large group photos in which every face is tiny.
DETECTION_MAX_SIZE = 640

# The mirror trick: each face is also turned into numbers as its mirror image, and the two results
# are averaged. Measured on LFW (see CLAUDE.md), it helps our model a little when a person is
# registered with 1 photo, and costs about 3 ms per face.
# It changes every embedding, so after changing it, register everyone again from the photo folders.
MIRROR_TRICK = True

# A face only gets a name if its similarity to that person reaches this value. Chosen on the LFW face
# dataset with 100 registered people (see CLAUDE.md) as the lowest value that gives at most 0.4% of
# strangers a name. At 0.48, 96.3% of registered people (5 photos each) were named correctly and 0.3%
# of strangers got a name.
# Lower it if registered people often show up as "Unknown"; raise it if strangers get names.
MATCH_THRESHOLD = 0.48

# The camera to use until one is chosen in the window's Camera list. Cameras are numbered from 0 in
# the order Windows lists them. The window saves its choice in CAMERA_CHOICE_FILE.
CAMERA_INDEX = 0
CAMERA_CHOICE_FILE = PROJECT_DIR / "data" / "camera.txt"
# A person must be recognized in this many video frames in a row to be marked present.
FRAMES_TO_CONFIRM = 10
