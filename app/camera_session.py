"""The camera, registration and live attendance, for the web version (web_app.py).

A web page can't run a loop on the server: it asks for things one request at a time. So the server
keeps the camera going by itself, and remembers where things stand: is the camera on, which of the
five registration photos is next, who is present today. The page asks for that a few times a second
(status, below), and fetches the newest picture again and again (next_picture).

Three kinds of threads share a CameraSession:
  - the camera thread (CameraThread in app/webcam.py) opens and reads the camera;
  - the worker thread (work, below) takes the camera's newest picture, finds and names the faces,
    takes the registration photos, marks attendance, and keeps the newest labelled picture;
  - Flask's threads answer the page's requests.
Two locks keep them out of each other's way: face_work lets only one of them use the face models
at a time, and lock protects the state they all read and change.
"""

import math
import shutil
import threading
import time
import traceback
from datetime import datetime

import cv2
import numpy as np

from app import config
from app.attendance import AttendanceRegister, attendance_on
from app.cameras import camera_names
from app.face_database import build_database, count_photos
from app.face_engine import FaceEngine
from app.recognizer import UNKNOWN, draw_results, recognize_faces
from app.webcam import (
    CAMERA_HELP,
    POSES,
    CameraThread,
    chosen_camera,
    is_valid_name,
    opening_message,
    remember_camera,
    save_photo,
)

SECONDS_BEFORE_EACH_PHOTO = 3
JPEG_QUALITY = 80
NAME_RULES = (
    "Type the person's name first. It can't use any of these: < > : \" / \\ | ? *, "
    "and it can't end with a dot or a space."
)


class Refused(Exception):
    """Something the page asked for that can't be done; the message says why, for the page to show."""


class CameraSession:
    def __init__(self, engine: FaceEngine, database: dict[str, np.ndarray], list_cameras=camera_names):
        self.engine = engine
        self.database = database
        self.list_cameras = list_cameras  # a stand-in can be passed by the tests
        self.face_work = threading.Lock()
        self.lock = threading.Lock()
        self.picture_ready = threading.Condition(self.lock)  # tells next_picture a new picture is there

        self.camera_labels: list[str] = []
        self.camera_number = chosen_camera()
        self.mode = None  # None while the camera is off, otherwise "live" or "register"
        self.camera = None  # the CameraThread while the camera is on
        self.camera_start_time = 0.0
        self.worker = None
        self.worker_stop = None  # an Event that tells the worker to finish
        self.picture = None  # the newest labelled picture, as JPEG bytes
        self.picture_number = 0
        self.picture_sent = 0
        self.message = "Choose a camera, then start live attendance or add a person."
        self.error = None
        self.attendance = None
        self.person_name = ""
        self.photos_taken = 0
        self.next_photo_time = None
        self.faces_seen = 0
        self.find_cameras()

    # ---------------------------------------------------------------- what the page can ask for

    def find_cameras(self) -> None:
        """List the cameras by name, in the order OpenCV numbers them (see app/cameras.py)."""
        names = self.list_cameras() or [f"Camera {number}" for number in range(3)]
        with self.lock:
            self.camera_labels = [f"{number}: {name}" for number, name in enumerate(names)]
            if self.camera_number >= len(self.camera_labels):
                self.camera_number = 0  # the chosen camera isn't connected now (the saved choice is kept)

    def choose_camera(self, camera_number: int) -> None:
        with self.lock:
            if not 0 <= camera_number < len(self.camera_labels):
                raise Refused("There is no camera with that number. Click Refresh to list the cameras again.")
            self.camera_number = camera_number
            camera_on = self.mode is not None
        remember_camera(camera_number)
        if camera_on:  # carry on with the same job (live or registering) on the new camera
            self.stop_camera()
            self.start_camera()

    def start_live(self) -> None:
        if not self.database:
            raise Refused("No one is registered yet. Add a person first.")
        self.stop()
        with self.lock:
            self.mode = "live"
            self.attendance = AttendanceRegister(config.ATTENDANCE_FILE)
            self.message = f"Looking for {len(self.database)} registered people..."
            self.error = None
        self.start_camera()

    def start_registration(self, name: str) -> None:
        name = name.strip()
        if not is_valid_name(name):
            raise Refused(NAME_RULES)
        self.stop()
        with self.lock:
            self.mode = "register"
            self.person_name = name
            self.photos_taken = 0
            self.message = f"Taking 5 photos of {name}. Follow the instructions on the picture."
            self.error = None
        self.start_camera()

    def stop(self) -> None:
        """Turn the camera off. A registration that was under way keeps the photos taken so far."""
        with self.lock:
            mode, name, photos = self.mode, self.person_name, self.photos_taken
            self.mode = None  # read and cleared together, so two stops can't register a person twice
        self.stop_camera()
        if mode == "register" and photos:
            self.finish_registration(name)
        elif mode:
            with self.lock:
                self.message = "The camera is off."

    def close(self) -> None:
        """Turn the camera off and wait for it to be released (when the program ends)."""
        camera = self.camera
        self.stop()
        if camera is not None:
            camera.stop(wait_seconds=3)

    def rebuild_database(self) -> int:
        """Register everyone again from the photo folders. Returns how many people are registered."""
        with self.face_work:
            database = build_database(self.engine, config.PEOPLE_DIR, config.DATABASE_FILE)
        self.database = database  # the worker and the photo page use the new one from now on
        return len(database)

    def add_photos(self, name: str, images: list[np.ndarray]) -> str:
        """Save uploaded photos in the person's folder and register everyone again."""
        name = name.strip()
        if not is_valid_name(name):
            raise Refused(NAME_RULES)
        if not images:
            raise Refused("Choose at least one photo of the person (3 to 5 is best).")
        person_dir = config.PEOPLE_DIR / name
        had_folder = person_dir.is_dir()
        saved = [save_photo(person_dir, image) for image in images]
        self.rebuild_database()
        if name not in self.database:
            # Don't keep photos nobody can see: an unregistered person isn't in the list to delete.
            for path in saved:
                path.unlink(missing_ok=True)
            if not had_folder and not any(person_dir.iterdir()):
                person_dir.rmdir()
            raise Refused(f"No face was found in those photos, so {name} is not registered and they were not kept.")
        return self.registration_result(name)

    def delete_person(self, name: str) -> None:
        """Delete a person's photo folder and register everyone else again."""
        person_dir = config.PEOPLE_DIR / name
        # resolve() checks where the folder really is, so no name can reach outside data/people/.
        if (
            not is_valid_name(name)
            or not person_dir.is_dir()
            or person_dir.resolve().parent != config.PEOPLE_DIR.resolve()
        ):
            raise Refused(f"There is no registered person called {name}.")
        with self.lock:
            if self.mode == "register" and self.person_name == name:
                raise Refused(f"{name} is being registered right now. Stop the camera first.")
        shutil.rmtree(person_dir)
        self.rebuild_database()

    def next_picture(self, timeout: float = 1.0) -> bytes | None:
        """The newest labelled picture, as JPEG bytes, or None while there is none. Waits up to
        `timeout` seconds for a picture newer than the last one sent, so each one is sent once."""
        with self.picture_ready:
            self.picture_ready.wait_for(
                lambda: self.picture is not None and self.picture_number != self.picture_sent, timeout
            )
            if self.picture is None:
                return None
            self.picture_sent = self.picture_number
            return self.picture

    def status(self) -> dict:
        """Everything the page shows."""
        today = datetime.now().strftime("%Y-%m-%d")
        with self.lock:
            camera_state = self.camera.state if self.camera else "off"
            message = self.message
            if camera_state == "opening":
                seconds = time.monotonic() - self.camera_start_time
                message = opening_message(self.label(self.camera.camera_number), seconds)
            registration = None
            if self.mode == "register":
                seconds_left = None  # the countdown starts with the camera's first picture
                if self.next_photo_time is not None:
                    seconds_left = max(0, math.ceil(self.next_photo_time - time.monotonic()))
                registration = {
                    "name": self.person_name,
                    "photos_taken": self.photos_taken,
                    "total": len(POSES),
                    "pose": POSES[min(self.photos_taken, len(POSES) - 1)],
                    "seconds_left": seconds_left,
                    "faces_seen": self.faces_seen,
                }
            status = {
                "mode": self.mode,
                "camera_state": camera_state,
                "cameras": list(self.camera_labels),
                "camera_number": self.camera_number,
                "message": message,
                "error": self.error,
                "registration": registration,
            }
        database = self.database
        status["people"] = [
            {"name": name, "photos": count_photos(config.PEOPLE_DIR / name)} for name in sorted(database)
        ]
        status["present_today"] = attendance_on(today, config.ATTENDANCE_FILE)
        status["today"] = today
        return status

    # ---------------------------------------------------------------- the camera and the worker

    def label(self, camera_number: int) -> str:
        if camera_number < len(self.camera_labels):
            return self.camera_labels[camera_number]
        return str(camera_number)

    def start_camera(self) -> None:
        """Start the camera thread and the worker, for the job in self.mode, with the chosen camera."""
        stop = threading.Event()
        with self.lock:
            self.camera = CameraThread(self.camera_number)
            self.camera_start_time = time.monotonic()
            self.next_photo_time = None  # registration's countdown starts with the camera's first picture
            self.worker_stop = stop
            self.worker = threading.Thread(target=self.work, args=(self.camera, stop), name="web worker", daemon=True)
            worker = self.worker
        worker.start()

    def stop_camera(self) -> None:
        """Stop the worker and the camera thread. (stop() also ends the job; this only stops the camera.)"""
        with self.lock:
            camera, stop, worker = self.camera, self.worker_stop, self.worker
            self.camera = self.worker = self.worker_stop = None
            if stop is not None:
                stop.set()  # set while holding the lock, so the worker can't show a picture after this
            self.picture = None
            self.picture_ready.notify_all()
        if camera is not None:
            camera.stop()  # its thread releases the camera; a new camera thread waits for that
        if worker is not None and worker is not threading.current_thread():
            worker.join(3)

    def work(self, camera: CameraThread, stop: threading.Event) -> None:
        """The worker thread: handle the camera's newest picture, again and again, until stopped."""
        try:
            while not stop.is_set():
                if camera.state in ("could not open", "could not read"):
                    with self.lock:
                        self.error = CAMERA_HELP if camera.state == "could not open" else "Could not read a picture from the camera."
                    self.stop()
                    return
                frame = camera.take_newest_frame()
                if frame is None:
                    stop.wait(0.01)  # no new picture yet
                    continue
                with self.lock:
                    mode = self.mode
                if mode == "live":
                    self.live_step(frame)
                elif mode == "register" and self.registration_step(frame):
                    return  # the fifth photo was taken
                self.publish(frame, stop)
        except Exception:  # anything unexpected: say so on the page, and print the details
            traceback.print_exc()
            with self.lock:
                self.error = "Something went wrong with the camera. The details are in the program's terminal."
            self.stop()

    def live_step(self, frame: np.ndarray) -> None:
        with self.face_work:
            results = recognize_faces(self.engine, self.database, frame)
        draw_results(frame, results)
        newly_present = self.attendance.update([result.name for result in results if result.name != UNKNOWN])
        if newly_present:
            with self.lock:
                self.message = f"Marked present at {datetime.now():%H:%M:%S}: {', '.join(newly_present)}"

    def registration_step(self, frame: np.ndarray) -> bool:
        """Take a photo when the countdown is over and exactly one face is visible. True after the last one."""
        with self.face_work:
            faces = self.engine.detect_faces(frame)
        now = time.monotonic()
        with self.lock:
            if self.next_photo_time is None:
                self.next_photo_time = now + SECONDS_BEFORE_EACH_PHOTO
            take_photo = now >= self.next_photo_time and len(faces) == 1
            self.faces_seen = len(faces)
            name = self.person_name
        if take_photo:
            save_photo(config.PEOPLE_DIR / name, frame)  # the clean picture, before boxes are drawn on it
            with self.lock:
                self.photos_taken += 1
                finished = self.photos_taken == len(POSES)
                self.next_photo_time = now + SECONDS_BEFORE_EACH_PHOTO
            if finished:
                self.stop()  # registers the person
                return True
        for face in faces:
            x, y, width, height = (int(value) for value in face[:4])
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 180, 0), 2)
        return False

    def finish_registration(self, name: str) -> None:
        with self.lock:
            self.message = f"Registering {name}..."
        self.rebuild_database()
        message = self.registration_result(name)
        with self.lock:
            self.message = message

    def registration_result(self, name: str) -> str:
        photos = count_photos(config.PEOPLE_DIR / name)
        if name not in self.database:
            return f"No face was found in {name}'s photos, so {name} is not registered."
        tip = " Add more photos of them: 3 to 5 is best." if photos < 3 else ""
        return f"{name} is registered with {photos} photo(s).{tip}"

    def publish(self, frame: np.ndarray, stop: threading.Event) -> None:
        """Keep the labelled picture as a JPEG, for next_picture."""
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with self.picture_ready:
            if stop.is_set():
                return  # the camera was turned off while this picture was being made
            self.picture = jpeg.tobytes()
            self.picture_number += 1
            self.picture_ready.notify_all()
