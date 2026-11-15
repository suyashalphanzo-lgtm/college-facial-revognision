"""Webcam features: add a new person (menu option 1) and live recognition with attendance (menu option 4).
Also opening the webcam, and the camera thread the window (ui.py) uses.
"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app import config
from app.attendance import AttendanceRegister
from app.face_database import build_database, load_database
from app.face_engine import FaceEngine
from app.recognizer import UNKNOWN, draw_results, recognize_faces

# On Windows, OpenCV can reach a webcam in two ways ("backends"):
#   DirectShow opens most webcams in about a second, at 640x480.
#   Media Foundation is OpenCV's own first choice and works with some newer laptop cameras that
#   DirectShow can't use. But on some computers it is very slow to open a camera, because it first
#   loads the graphics card's video converters. The setting below turns those off; OpenCV reads it
#   every time it opens a camera.
# So on Windows DirectShow is tried first, then Media Foundation. Elsewhere OpenCV chooses.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
CAMERA_BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY]
# A backend only counts as working once it gives a picture. Some webcams need a moment for the first one.
SECONDS_TO_WAIT_FOR_FIRST_PICTURE = 5
CAMERA_HELP = (
    "Could not open the webcam. Check that no other app is using it (Camera, Zoom, Teams, a browser tab), "
    "that Windows Settings > Privacy & security > Camera lets desktop apps use the camera, and that the "
    "laptop's camera switch or camera key isn't turning it off. Or choose another camera in the window's "
    "Camera list."
)

WINDOW_TITLE = "Webcam (press Q to quit)"
SPACE_KEY = 32
ESCAPE_KEY = 27
INVALID_NAME_CHARACTERS = set('<>:"/\\|?*')
# One photo is taken for each pose, so the person can later be recognized from different angles.
POSES = [
    "Look straight at the camera",
    "Turn your head slightly left",
    "Turn your head slightly right",
    "Tilt your head slightly up",
    "Tilt your head slightly down",
]


def is_valid_name(name: str) -> bool:
    """A name becomes a folder name, so it can't be empty or use characters Windows forbids in folder
    names. It can't end with a dot or a space either: Windows drops those, so "Aadi." would become the
    folder "Aadi", and "." or ".." would point at a folder that isn't a person's."""
    return (
        bool(name)
        and name.isprintable()
        and not INVALID_NAME_CHARACTERS & set(name)
        and not name.endswith((".", " "))
    )


def opening_message(camera_label: str, seconds: float) -> str:
    """What to show while a camera is opening (the window and the web page both use it)."""
    if seconds < 3:
        return f"Opening camera {camera_label}..."
    if seconds < 20:
        return f"Opening camera {camera_label}... ({seconds:.0f} s)"
    return (
        f"Camera {camera_label} still hasn't opened after {seconds:.0f} seconds. Check that no other app is "
        "using it, or choose another camera. If it never opens, stop the camera, then close this program "
        "and start it again."
    )


def chosen_camera() -> int:
    """The camera chosen in the window's Camera list, or CAMERA_INDEX if none was chosen yet."""
    try:
        return int(config.CAMERA_CHOICE_FILE.read_text().strip())
    except (OSError, ValueError):
        return config.CAMERA_INDEX


def remember_camera(camera_number: int) -> None:
    config.CAMERA_CHOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CAMERA_CHOICE_FILE.write_text(str(camera_number))


def save_photo(person_dir: Path, frame: np.ndarray) -> Path:
    """Save a photo in a person's folder, named after the moment it was taken."""
    person_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = person_dir / f"{stem}.jpg"
    number = 1
    while path.exists():  # several uploaded photos can be saved within the same clock tick
        path = person_dir / f"{stem}_{number}.jpg"
        number += 1
    cv2.imwrite(str(path), frame)
    return path


def add_person(engine: FaceEngine, name: str) -> None:
    """Take one photo per pose, save them in data/people/<name>/, then rebuild the database."""
    if not is_valid_name(name):
        print('Please type a name that does not use any of these characters: < > : " / \\ | ? *')
        return
    camera = open_camera(chosen_camera())
    if camera is None:
        return

    person_dir = config.PEOPLE_DIR / name
    photos_saved = 0
    print("Follow the instruction at the top of the window and press SPACE for each photo.")
    print("Press Q to stop early.")
    while photos_saved < len(POSES):
        frame = read_frame(camera)
        if frame is None:
            break
        faces = engine.detect_faces(frame)

        preview = frame.copy()
        for face in faces:
            x, y, width, height = (int(value) for value in face[:4])
            cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 180, 0), 2)
        instruction = f"Photo {photos_saved + 1}/{len(POSES)}: {POSES[photos_saved]}"
        cv2.putText(preview, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(preview, "SPACE = take photo, Q = stop", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WINDOW_TITLE, preview)

        key = cv2.waitKey(1) & 0xFF
        if key == SPACE_KEY:
            if len(faces) == 1:
                save_photo(person_dir, frame)
                photos_saved += 1
                print(f"  Saved photo {photos_saved}/{len(POSES)}")
            else:
                print(f"  Not saved: {len(faces)} faces visible, there must be exactly 1")
        elif should_quit(key):
            break
    camera.release()
    cv2.destroyAllWindows()

    if photos_saved:
        build_database(engine)


def recognize_live(engine: FaceEngine) -> None:
    """Name every face the webcam sees and mark attendance."""
    database = load_database()
    if not database:
        print("No one is registered yet. Add people first (menu option 1 or 2).")
        return
    camera = open_camera(chosen_camera())
    if camera is None:
        return

    attendance = AttendanceRegister()
    print(f"Looking for {len(database)} registered people. Press Q to stop.")
    while True:
        frame = read_frame(camera)
        if frame is None:
            break
        results = recognize_faces(engine, database, frame)
        recognized_names = [result.name for result in results if result.name != UNKNOWN]
        for name in attendance.update(recognized_names):
            print(f"  {name} marked present at {datetime.now():%H:%M:%S}")

        draw_results(frame, results)
        cv2.imshow(WINDOW_TITLE, frame)
        if should_quit(cv2.waitKey(1) & 0xFF):
            break
    camera.release()
    cv2.destroyAllWindows()
    print(f"Present today: {len(attendance.present_today)}. Attendance is saved in {config.ATTENDANCE_FILE}")


def open_camera(camera_number: int) -> cv2.VideoCapture | None:
    """Open a camera with the first backend that gives a picture."""
    print(f"Opening camera {camera_number}...")
    for backend in CAMERA_BACKENDS:
        camera = cv2.VideoCapture(camera_number, backend)
        if camera.isOpened() and gives_a_picture(camera):
            return camera
        camera.release()
    print(CAMERA_HELP)
    return None


def gives_a_picture(camera: cv2.VideoCapture) -> bool:
    give_up_time = time.monotonic() + SECONDS_TO_WAIT_FOR_FIRST_PICTURE
    while time.monotonic() < give_up_time:
        if camera.read()[0]:
            return True
        time.sleep(0.1)
    return False


def read_frame(camera: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = camera.read()
    if not ok:
        print("Could not read a picture from the webcam.")
        return None
    return cv2.flip(frame, 1)  # mirror the picture, so moving left also moves left on screen


def should_quit(key: int) -> bool:
    window_closed = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
    return key in (ord("q"), ord("Q"), ESCAPE_KEY) or window_closed


class CameraThread:
    """The webcam for the window (ui.py), opened and read in a thread of its own.

    Opening a webcam can take seconds, and every read waits for the camera's next picture. While Python
    waits, a Tkinter window can't redraw itself or react to its buttons, and Windows marks it
    "Not Responding". So this thread does all the waiting, and the window takes the newest picture
    whenever it is ready for one. Only this thread touches the camera: it opens, reads and releases it.
    """

    # A webcam can only be opened once at a time, so a new CameraThread waits until the previous one
    # has released it (the window doesn't wait for that when the camera is stopped).
    webcam_in_use = threading.Lock()

    def __init__(self, camera_number: int) -> None:
        self.camera_number = camera_number
        self.state = "opening"  # then "on"; at the end "off", "could not open" or "could not read"
        self.newest_frame = None
        self.frame_lock = threading.Lock()  # the two threads take turns with newest_frame
        self.stop_requested = threading.Event()
        # daemon: a camera that never finishes opening can't keep Python running after the window closes
        self.thread = threading.Thread(target=self.run, name="webcam", daemon=True)
        self.thread.start()

    def run(self) -> None:
        with CameraThread.webcam_in_use:
            try:
                self.open_and_read()
            finally:
                # If OpenCV raised an error (Python prints it), the window must still learn that the
                # camera isn't running. Otherwise it would show "Opening the webcam..." forever.
                if self.state == "opening" and not self.stop_requested.is_set():
                    self.state = "could not open"
                elif self.state == "on" and not self.stop_requested.is_set():
                    self.state = "could not read"

    def open_and_read(self) -> None:
        camera = None if self.stop_requested.is_set() else open_camera(self.camera_number)
        if camera is None:
            self.state = "off" if self.stop_requested.is_set() else "could not open"
            return
        self.state = "on"
        try:
            while not self.stop_requested.is_set():
                frame = read_frame(camera)
                if frame is None:
                    self.state = "could not read"
                    return
                with self.frame_lock:
                    self.newest_frame = frame
        finally:
            camera.release()
        self.state = "off"

    def take_newest_frame(self) -> np.ndarray | None:
        """The newest picture, or None if no new picture has arrived since the last call."""
        with self.frame_lock:
            frame, self.newest_frame = self.newest_frame, None
        return frame

    def stop(self, wait_seconds: float = 0) -> None:
        """Ask the thread to turn the webcam off. With wait_seconds, wait up to that long for it."""
        self.stop_requested.set()
        if wait_seconds:
            self.thread.join(wait_seconds)
