from app import config
from app.attendance import AttendanceRegister


def show_frames(register, names, frame_count):
    """Pretend the webcam saw the same names in frame_count frames; return who got marked present."""
    marked = []
    for _ in range(frame_count):
        marked += register.update(names)
    return marked


def test_marked_present_after_enough_frames_in_a_row(tmp_path):
    register = AttendanceRegister(tmp_path / "attendance.csv")
    assert show_frames(register, ["Asha"], config.FRAMES_TO_CONFIRM - 1) == []
    assert show_frames(register, ["Asha"], 1) == ["Asha"]


def test_count_restarts_when_the_person_leaves_the_frame(tmp_path):
    register = AttendanceRegister(tmp_path / "attendance.csv")
    show_frames(register, ["Asha"], config.FRAMES_TO_CONFIRM - 1)
    show_frames(register, [], 1)
    assert show_frames(register, ["Asha"], config.FRAMES_TO_CONFIRM - 1) == []


def test_marked_only_once_per_day_even_after_a_restart(tmp_path):
    csv_file = tmp_path / "attendance.csv"
    show_frames(AttendanceRegister(csv_file), ["Asha"], config.FRAMES_TO_CONFIRM)
    assert show_frames(AttendanceRegister(csv_file), ["Asha"], config.FRAMES_TO_CONFIRM) == []

    rows = csv_file.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "name,date,time"
    assert len(rows) == 2
