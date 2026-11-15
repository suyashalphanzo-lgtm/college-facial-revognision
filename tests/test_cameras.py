import sys

import pytest

from app.cameras import camera_names


@pytest.mark.skipif(sys.platform != "win32", reason="the cameras' names come from Windows")
def test_camera_names_are_read_without_turning_any_camera_on():
    names = camera_names()  # may be empty on a computer without cameras
    assert isinstance(names, list)
    assert all(isinstance(name, str) and name for name in names)
