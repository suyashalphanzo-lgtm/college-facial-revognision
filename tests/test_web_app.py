import io
import socket
import threading
import time

import cv2
import numpy as np
import pytest

import web_app
from app import camera_session, config, webcam
from web_app import create_app

FROM_THE_PAGE = {"X-Face-App": "1"}  # the header the page's JavaScript adds to every request


class FakeEngine:
    """Stands in for the face models: finds one face in the middle of every picture, always Asha's."""

    def detect_faces(self, image):
        height, width = image.shape[:2]
        return [np.array([width // 4, height // 4, width // 2, height // 2] + [0] * 10 + [0.9], np.float32)]

    def get_embedding(self, image, face):
        return np.array([0.6, 0.8], np.float32)


class FakeCamera:
    """Stands in for cv2.VideoCapture: grey pictures, about 100 a second. No real camera is opened."""

    def __init__(self, index, backend):
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.01)
        if self.released:
            return False, None
        return True, np.full((120, 160, 3), 128, np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def site(tmp_path, monkeypatch):
    for setting, name in [
        ("PEOPLE_DIR", "people"),
        ("DATABASE_FILE", "face_database_ours.json"),
        ("ATTENDANCE_FILE", "attendance.csv"),
        ("RESULTS_DIR", "results"),
        ("CAMERA_CHOICE_FILE", "camera.txt"),
    ]:
        monkeypatch.setattr(config, setting, tmp_path / name)
    monkeypatch.setattr(webcam.cv2, "VideoCapture", FakeCamera)
    monkeypatch.setattr(webcam, "CAMERA_BACKENDS", ["fake"])
    monkeypatch.setattr(camera_session, "SECONDS_BEFORE_EACH_PHOTO", 0.05)
    web = create_app(
        FakeEngine(), {"Asha": np.array([0.6, 0.8], np.float32)}, list_cameras=lambda: ["Fake Camera", "Second Camera"]
    )
    yield web
    web.extensions["camera_session"].close()
    for thread in threading.enumerate():  # let the camera threads end before the real VideoCapture comes back
        if thread.name in ("webcam", "web worker"):
            thread.join(2)


@pytest.fixture
def browser(site):
    """Flask's test client: asks the site for things the way the browser would, without a real browser."""
    return site.test_client()


def status(browser) -> dict:
    return browser.get("/api/status").get_json()


def send(browser, path: str, **json):
    return browser.post(path, json=json, headers=FROM_THE_PAGE)


def photo_file(name: str = "photo.png"):
    _, png = cv2.imencode(".png", np.full((60, 80, 3), 128, np.uint8))
    return io.BytesIO(png.tobytes()), name


def wait_until(condition, seconds: float = 5.0) -> None:
    give_up_time = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < give_up_time, "waited too long"
        time.sleep(0.02)


def test_the_page_and_its_status(browser):
    assert "College Face Recognition" in browser.get("/").get_data(as_text=True)
    current = status(browser)
    assert current["mode"] is None
    assert current["cameras"] == ["0: Fake Camera", "1: Second Camera"]
    assert [person["name"] for person in current["people"]] == ["Asha"]


def test_changes_are_only_accepted_from_the_page_itself(browser):
    assert browser.post("/api/live/start").status_code == 403  # no header: like another website trying
    assert browser.get("/api/status", base_url="http://evil.example").status_code == 400  # not this computer
    assert status(browser)["mode"] is None


def test_an_uploaded_photo_comes_back_labelled_and_is_saved(browser, tmp_path):
    response = browser.post("/api/recognize", data={"photo": photo_file("class photo.png")}, headers=FROM_THE_PAGE)
    answer = response.get_json()
    assert answer["image"].startswith("data:image/jpeg;base64,")  # the labelled photo, inside the answer
    assert answer["faces"] == [{"name": "Asha", "similarity": 1.0, "known": True}]
    assert (tmp_path / "results" / "class_photo_result.jpg").exists()  # a safe name for the saved copy


def test_a_file_that_is_not_a_photo_is_refused(browser):
    response = browser.post("/api/recognize", data={"photo": (io.BytesIO(b"just text"), "notes.txt")}, headers=FROM_THE_PAGE)
    assert response.status_code == 400
    assert "Could not read that file as a photo" in response.get_json()["error"]


def test_the_live_camera_marks_registered_people_present(browser):
    assert send(browser, "/api/live/start").status_code == 200
    wait_until(lambda: status(browser)["present_today"])  # recognized in 10 pictures in a row
    assert [arrival["name"] for arrival in status(browser)["present_today"]] == ["Asha"]

    picture = browser.get("/api/frame.jpg")
    assert picture.status_code == 200 and picture.mimetype == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(picture.data, np.uint8), cv2.IMREAD_COLOR) is not None

    send(browser, "/api/stop")
    assert status(browser)["mode"] is None
    assert browser.get("/api/frame.jpg").status_code == 204  # no picture while the camera is off


def test_live_attendance_needs_someone_registered(site, browser):
    site.extensions["camera_session"].database = {}
    response = send(browser, "/api/live/start")
    assert response.status_code == 400
    assert "No one is registered yet" in response.get_json()["error"]


def test_registration_takes_five_photos_and_registers_the_person(browser, tmp_path):
    assert send(browser, "/api/register/start", name="Priya").status_code == 200
    # After the fifth photo the camera turns off, and then everyone is registered again.
    wait_until(lambda: "Priya" in [person["name"] for person in status(browser)["people"]])
    assert len(list((tmp_path / "people" / "Priya").glob("*.jpg"))) == 5
    current = status(browser)
    assert current["mode"] is None
    assert current["message"] == "Priya is registered with 5 photo(s)."


@pytest.mark.parametrize("name", ["", "..", "a/b", "Aadi."])
def test_names_that_cant_be_folder_names_are_refused(browser, name):
    assert send(browser, "/api/register/start", name=name).status_code == 400
    assert status(browser)["mode"] is None


def test_a_person_can_be_registered_from_uploaded_photos(browser, tmp_path):
    photos = [photo_file("1.png"), photo_file("2.png"), photo_file("3.png")]
    response = browser.post("/api/people/upload", data={"name": "Ravi", "photos": photos}, headers=FROM_THE_PAGE)
    assert response.get_json()["message"] == "Ravi is registered with 3 photo(s)."
    assert len(list((tmp_path / "people" / "Ravi").glob("*.jpg"))) == 3


def test_photos_without_a_face_are_not_kept(site, browser, tmp_path):
    site.extensions["camera_session"].engine.detect_faces = lambda image: []  # a camera roll with no faces
    response = browser.post("/api/people/upload", data={"name": "Ravi", "photos": [photo_file()]}, headers=FROM_THE_PAGE)
    assert response.status_code == 400
    assert "No face was found" in response.get_json()["error"]
    assert not (tmp_path / "people" / "Ravi").exists()


def test_a_person_can_be_deleted(browser, tmp_path):
    browser.post("/api/people/upload", data={"name": "Ravi", "photos": [photo_file()]}, headers=FROM_THE_PAGE)
    assert send(browser, "/api/people/delete", name="Ravi").status_code == 200
    assert not (tmp_path / "people" / "Ravi").exists()
    assert "Ravi" not in [person["name"] for person in status(browser)["people"]]


def test_only_registered_people_can_be_deleted(browser, tmp_path):
    (tmp_path / "people").mkdir()
    (tmp_path / "keep").mkdir()
    for name in ["..", "../keep", "Nobody", ""]:
        assert send(browser, "/api/people/delete", name=name).status_code == 400
    assert (tmp_path / "keep").exists()


def test_attendance_for_one_date_and_as_a_csv_file(browser, tmp_path):
    (tmp_path / "attendance.csv").write_text(
        "name,date,time\nAsha,2026-11-14,09:01:00\nRavi,2026-11-15,09:05:00\nAsha,2026-11-15,09:07:00\n",
        encoding="utf-8",
    )
    answer = browser.get("/api/attendance?date=2026-11-15").get_json()
    assert answer["rows"] == [{"name": "Ravi", "time": "09:05:00"}, {"name": "Asha", "time": "09:07:00"}]
    assert answer["dates"] == ["2026-11-15", "2026-11-14"]

    download = browser.get("/attendance.csv?date=2026-11-14")
    assert download.headers["Content-Disposition"] == "attachment; filename=attendance_2026-11-14.csv"
    assert download.get_data(as_text=True).splitlines() == ["name,date,time", "Asha,2026-11-14,09:01:00"]
    assert browser.get("/api/attendance?date=yesterday").status_code == 400


def test_the_chosen_camera_is_remembered(browser, tmp_path):
    assert status(browser)["camera_number"] == 0
    assert send(browser, "/api/camera", camera=1).status_code == 200
    assert status(browser)["camera_number"] == 1
    assert (tmp_path / "camera.txt").read_text() == "1"
    assert send(browser, "/api/camera", camera=7).status_code == 400


def test_choosing_another_camera_keeps_the_live_camera_running(browser):
    send(browser, "/api/live/start")
    send(browser, "/api/camera", camera=1)
    current = status(browser)
    assert current["mode"] == "live"
    assert current["camera_number"] == 1


def test_a_second_copy_does_not_start_while_one_is_running(monkeypatch):
    monkeypatch.setattr(web_app, "FaceEngine", lambda: pytest.fail("the models were loaded"))
    with socket.create_server(("127.0.0.1", 0)) as first_copy:  # stands in for a copy that is already running
        monkeypatch.setattr(web_app, "PORT", first_copy.getsockname()[1])
        assert web_app.already_running()
        with pytest.raises(SystemExit, match="already running"):
            web_app.main()  # stops before loading the models
    assert not web_app.already_running()  # the first copy has stopped: the port is free again
