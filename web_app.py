"""College face recognition system in the browser. Start it with:  python web_app.py

A small website that runs only on this computer. Flask, a Python web framework, answers the browser:
each address (a "route"), like "/" or "/api/status", is one of the Python functions below. The page
is web/templates/index.html, its look web/static/style.css, and its behaviour web/static/app.js,
the JavaScript that runs in the browser and asks these routes for things:

  GET  /                     the page
  GET  /api/status           everything the page shows: camera, registration, present today, people
  GET  /api/frame.jpg        the camera's newest picture, with every face boxed and named
  POST /api/cameras/refresh  list the cameras again (after plugging one in)
  POST /api/camera           choose a camera
  POST /api/live/start       start live recognition and attendance
  POST /api/register/start   take the 5 registration photos of a person with the camera
  POST /api/stop             turn the camera off
  POST /api/recognize        name the faces in an uploaded photo
  POST /api/people/upload    register a person from uploaded photos
  POST /api/people/rebuild   register everyone again from the photo folders
  POST /api/people/delete    delete a person and their photos
  GET  /api/attendance       who was present on a date
  GET  /attendance.csv       download the attendance (one date, or everything) as a CSV file

The camera, registration and attendance work is in app/camera_session.py; the face work in app/.
"""

import base64
import csv
import io
import socket
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request
from werkzeug.utils import secure_filename

from app import config
from app.attendance import attendance_dates, attendance_on, read_attendance
from app.camera_session import CameraSession, Refused
from app.cameras import camera_names
from app.face_database import load_or_build_database
from app.face_engine import FaceEngine
from app.photo import label_and_save
from app.recognizer import UNKNOWN

# 127.0.0.1 means "this computer only": no other device, even on the same Wi-Fi, can open the site.
ADDRESS, PORT = "127.0.0.1", 5000
THIS_COMPUTER = {"127.0.0.1", "localhost"}
LARGEST_UPLOAD = 50 * 1024 * 1024  # bytes, all files of one upload together; bigger uploads are refused
SHOWN_WIDTH = 960  # a recognized photo is shrunk to this width for the page (the saved copy is full size)


def create_app(engine: FaceEngine, database: dict[str, np.ndarray], list_cameras=camera_names) -> Flask:
    """Make the website. The models, the database and the camera list are passed in, so the tests can
    pass stand-ins."""
    web = Flask(__name__, template_folder="web/templates", static_folder="web/static")
    web.config["MAX_CONTENT_LENGTH"] = LARGEST_UPLOAD
    session = CameraSession(engine, database, list_cameras)
    web.extensions["camera_session"] = session  # so main() and the tests can reach it

    @web.before_request
    def only_this_computer_and_this_page():
        # Only addresses that mean this computer. Otherwise a web page elsewhere could use a domain
        # name that points at 127.0.0.1 to read this site ("DNS rebinding").
        if request.host.split(":")[0] not in THIS_COMPUTER:
            abort(400)
        # Changes only from our own page: its JavaScript adds this header to every request. Other
        # websites open in the same browser can't add it, so they can't press this site's buttons.
        if request.method == "POST" and request.headers.get("X-Face-App") != "1":
            abort(403)

    @web.errorhandler(Refused)
    def refused(problem: Refused):
        return jsonify(error=str(problem)), 400

    @web.errorhandler(413)
    def too_big(error):
        return jsonify(error=f"That is more than {LARGEST_UPLOAD // (1024 * 1024)} MB of photos at once."), 413

    @web.get("/")
    def page():
        return render_template("index.html")

    @web.get("/api/status")
    def status():
        return jsonify(session.status())

    @web.get("/api/frame.jpg")
    def frame():
        picture = session.next_picture()
        if picture is None:
            return Response(status=204)  # no picture yet: the camera is off or still opening
        return Response(picture, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @web.post("/api/cameras/refresh")
    def refresh_cameras():
        session.find_cameras()
        return jsonify(session.status())

    @web.post("/api/camera")
    def choose_camera():
        camera_number = sent_json().get("camera")
        if not isinstance(camera_number, int):
            raise Refused("Choose a camera from the list.")
        session.choose_camera(camera_number)
        return jsonify(session.status())

    @web.post("/api/live/start")
    def start_live():
        session.start_live()
        return jsonify(session.status())

    @web.post("/api/register/start")
    def start_registration():
        session.start_registration(str(sent_json().get("name", "")))
        return jsonify(session.status())

    @web.post("/api/stop")
    def stop():
        session.stop()
        return jsonify(session.status())

    @web.post("/api/recognize")
    def recognize():
        upload = request.files.get("photo")
        image = read_image(upload)
        if image is None:
            raise Refused("Could not read that file as a photo. Choose a JPG, PNG, BMP or WEBP photo.")
        file_name = secure_filename(upload.filename) or "photo.jpg"  # a safe file name for the saved copy
        with session.face_work:
            results, result_path = label_and_save(session.engine, session.database, image, Path(file_name))
        return jsonify(
            file_name=upload.filename,
            image=as_data_url(image),
            faces=[
                {"name": face.name, "similarity": round(face.similarity, 2), "known": face.name != UNKNOWN}
                for face in results
            ],
            saved_as=f"data/results/{result_path.name}",
        )

    @web.post("/api/people/upload")
    def upload_photos():
        images = [read_image(upload) for upload in request.files.getlist("photos")]
        photos = [image for image in images if image is not None]
        message = session.add_photos(request.form.get("name", ""), photos)
        if len(photos) < len(images):
            message += f" {len(images) - len(photos)} file(s) could not be read as photos."
        return jsonify(message=message)

    @web.post("/api/people/rebuild")
    def rebuild():
        return jsonify(message=f"People registered: {session.rebuild_database()}.")

    @web.post("/api/people/delete")
    def delete_person():
        name = str(sent_json().get("name", ""))
        session.delete_person(name)
        return jsonify(message=f"{name} and their photos are deleted.")

    @web.get("/api/attendance")
    def attendance():
        date = checked_date(request.args.get("date") or datetime.now().strftime("%Y-%m-%d"))
        return jsonify(
            date=date,
            rows=attendance_on(date, config.ATTENDANCE_FILE),
            dates=attendance_dates(config.ATTENDANCE_FILE),
        )

    @web.get("/attendance.csv")
    def attendance_csv():
        rows = read_attendance(config.ATTENDANCE_FILE)
        date = request.args.get("date")
        if date:
            rows = [row for row in rows if row["date"] == checked_date(date)]
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=["name", "date", "time"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        file_name = f"attendance_{date}.csv" if date else "attendance.csv"
        return Response(
            text.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={file_name}"},  # download, don't show
        )

    return web


def sent_json() -> dict:
    """The JSON the page sent with its request, or an empty dictionary."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def checked_date(date: str) -> str:
    """A date written like 2026-11-15; anything else is refused."""
    try:
        return datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise Refused("Choose a date.") from None


def read_image(upload) -> np.ndarray | None:
    """An uploaded file as an OpenCV image, or None if it isn't a photo OpenCV can read."""
    if upload is None or not upload.filename:
        return None
    data = np.frombuffer(upload.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def as_data_url(image: np.ndarray) -> str:
    """The image as text the page can show directly: <img src="data:image/jpeg;base64,...">.
    That way the labelled photo doesn't have to be saved anywhere the browser can fetch it from."""
    scale = min(1.0, SHOWN_WIDTH / image.shape[1])
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")


def already_running() -> bool:
    """True when something already answers at ADDRESS:PORT, most likely this web version, started earlier.

    Flask's server asks to share its port (SO_REUSEADDR), and on Windows that lets a second copy start on
    the same port without an error. The browser then goes on talking to the first copy, old code and all.
    On Windows a closed port takes 2 seconds to refuse, so this waits at most half a second."""
    try:
        with socket.create_connection((ADDRESS, PORT), timeout=0.5):
            return True
    except OSError:  # refused, or no answer in time: nothing is running there
        return False


def main() -> None:
    address = f"http://{ADDRESS}:{PORT}"
    if already_running():
        raise SystemExit(
            f"The web version is already running at {address} (or another program is using that address).\n"
            "To use it, open that address in your browser. To start it again, for example after changing the\n"
            "code, first press Ctrl+C in the terminal where it runs, or close that terminal."
        )
    print("Loading the face models (YuNet finds faces, our own SFace recognizes them)...")
    engine = FaceEngine()
    database = load_or_build_database(engine)  # registers everyone first if there is no database yet
    web = create_app(engine, database)
    print(f"Opening {address} in your browser. Keep this window open; press Ctrl+C here to stop.")
    threading.Timer(1.0, webbrowser.open, args=(address,)).start()  # once the server is up
    try:
        web.run(host=ADDRESS, port=PORT, debug=False, threaded=True)  # debug mode would let a page run Python code
    finally:
        web.extensions["camera_session"].close()  # turn the camera off properly


if __name__ == "__main__":
    main()
