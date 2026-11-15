import threading
import time

import numpy as np
import pytest

from app import config, webcam
from app.webcam import CameraThread, chosen_camera, is_valid_name, open_camera, remember_camera


def test_names_that_can_be_folder_names_are_accepted():
    assert is_valid_name("Aaditya Sharma")


def test_empty_names_and_characters_windows_forbids_are_rejected():
    assert not is_valid_name("")
    assert not is_valid_name("Aaditya/Sharma")
    assert not is_valid_name("Who?")


def test_names_windows_would_change_are_rejected():
    # Windows drops a dot or space at the end of a folder name, and "." and ".." aren't people's folders.
    for name in [".", "..", "Aadi.", "Aadi ", "Tab\there"]:
        assert not is_valid_name(name)
    assert is_valid_name("A. Sharma")


@pytest.fixture
def fake_webcam(monkeypatch):
    """Replaces OpenCV's VideoCapture with a fake one, so these tests never open a real webcam."""

    class FakeVideoCapture:
        working_backends = {"DirectShow"}
        seconds_to_open = 0.0
        made = []  # every fake camera made, oldest first

        def __init__(self, index, backend):
            time.sleep(FakeVideoCapture.seconds_to_open)
            self.index = index
            self.backend = backend
            self.webcam_was_free = all(camera.released for camera in FakeVideoCapture.made)
            self.released = False
            FakeVideoCapture.made.append(self)

        def isOpened(self):
            return self.backend in FakeVideoCapture.working_backends

        def read(self):
            time.sleep(0.01)
            if self.isOpened() and not self.released:
                return True, np.zeros((48, 64, 3), np.uint8)
            return False, None

        def release(self):
            self.released = True

    monkeypatch.setattr(webcam.cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(webcam, "CAMERA_BACKENDS", ["DirectShow", "Media Foundation"])
    monkeypatch.setattr(webcam, "SECONDS_TO_WAIT_FOR_FIRST_PICTURE", 0.2)
    yield FakeVideoCapture
    # Let every camera thread finish before the real VideoCapture comes back.
    for thread in threading.enumerate():
        if thread.name == "webcam":
            thread.join(2)


def wait_until(condition, seconds=2.0):
    give_up_time = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < give_up_time, "waited too long"
        time.sleep(0.01)


def test_open_camera_tries_the_next_backend_when_one_does_not_work(fake_webcam):
    fake_webcam.working_backends = {"Media Foundation"}
    camera = open_camera(0)
    assert camera.backend == "Media Foundation"
    first_try = fake_webcam.made[0]
    assert first_try.backend == "DirectShow" and first_try.released


def test_open_camera_gives_none_when_no_backend_works(fake_webcam):
    fake_webcam.working_backends = set()
    assert open_camera(0) is None
    assert len(fake_webcam.made) == 2
    assert all(camera.released for camera in fake_webcam.made)


def test_camera_thread_opens_the_webcam_without_making_the_window_wait(fake_webcam):
    fake_webcam.seconds_to_open = 0.3
    started = time.monotonic()
    camera = CameraThread(0)
    assert time.monotonic() - started < 0.1
    assert camera.state == "opening"
    wait_until(lambda: camera.take_newest_frame() is not None)
    assert camera.state == "on"
    camera.stop(wait_seconds=2)
    assert camera.state == "off"
    assert fake_webcam.made[-1].released


def test_camera_thread_reports_a_webcam_that_cannot_be_opened(fake_webcam):
    fake_webcam.working_backends = set()
    camera = CameraThread(0)
    wait_until(lambda: camera.state == "could not open")


def test_a_webcam_stopped_while_still_opening_is_released(fake_webcam):
    fake_webcam.seconds_to_open = 0.2
    camera = CameraThread(0)
    camera.stop(wait_seconds=2)
    assert camera.state == "off"
    assert all(camera.released for camera in fake_webcam.made)


def test_a_new_camera_thread_waits_until_the_old_one_has_released_the_webcam(fake_webcam):
    first = CameraThread(0)
    wait_until(lambda: first.state == "on")
    first.stop()  # without waiting, like the window's Stop camera button
    second = CameraThread(0)
    wait_until(lambda: second.state == "on")
    second.stop(wait_seconds=2)
    assert len(fake_webcam.made) == 2
    assert all(camera.webcam_was_free for camera in fake_webcam.made)


def test_camera_thread_opens_the_chosen_camera(fake_webcam):
    camera = CameraThread(2)
    wait_until(lambda: camera.state == "on")
    camera.stop(wait_seconds=2)
    assert [made.index for made in fake_webcam.made] == [2]


def test_the_chosen_camera_is_remembered(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAMERA_CHOICE_FILE", tmp_path / "camera.txt")
    assert chosen_camera() == config.CAMERA_INDEX  # nothing chosen yet

    remember_camera(1)
    assert chosen_camera() == 1

    (tmp_path / "camera.txt").write_text("not a number")
    assert chosen_camera() == config.CAMERA_INDEX
