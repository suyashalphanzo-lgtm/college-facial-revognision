"""Attendance register, saved in data/attendance.csv (it opens in Excel).

A person is marked present at most once per day, and only after being recognized in
config.FRAMES_TO_CONFIRM video frames in a row, so one wrong guess can't mark someone present.
"""

import csv
from datetime import datetime
from pathlib import Path

from app import config


class AttendanceRegister:
    def __init__(self, csv_file: Path = config.ATTENDANCE_FILE):
        self.csv_file = csv_file
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.present_today = self._read_present_today()
        self.frames_in_a_row: dict[str, int] = {}

    def _read_present_today(self) -> set[str]:
        if not self.csv_file.exists():
            return set()
        with self.csv_file.open(newline="", encoding="utf-8") as file:
            return {row["name"] for row in csv.DictReader(file) if row["date"] == self.today}

    def update(self, names_in_frame: list[str]) -> list[str]:
        """Call once per video frame with the names recognized in it. Returns who was just marked present."""
        # Anyone missing from this frame is left out of the new dictionary, so their count restarts.
        self.frames_in_a_row = {name: self.frames_in_a_row.get(name, 0) + 1 for name in set(names_in_frame)}
        newly_present = [
            name
            for name, count in self.frames_in_a_row.items()
            if count >= config.FRAMES_TO_CONFIRM and name not in self.present_today
        ]
        for name in newly_present:
            self.mark_present(name)
        return newly_present

    def mark_present(self, name: str) -> None:
        """Add one row to the CSV file: name, date, time."""
        is_new_file = not self.csv_file.exists()
        self.csv_file.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if is_new_file:
                writer.writerow(["name", "date", "time"])
            writer.writerow([name, self.today, datetime.now().strftime("%H:%M:%S")])
        self.present_today.add(name)


def read_attendance(csv_file: Path) -> list[dict[str, str]]:
    """Every row of the attendance file, oldest first: {"name": ..., "date": ..., "time": ...}."""
    if not csv_file.exists():
        return []
    with csv_file.open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if row.get("name") and row.get("date")]


def attendance_on(date: str, csv_file: Path) -> list[dict[str, str]]:
    """Who was marked present on one date (written like 2026-11-15), in the order they arrived."""
    return [{"name": row["name"], "time": row["time"]} for row in read_attendance(csv_file) if row["date"] == date]


def attendance_dates(csv_file: Path) -> list[str]:
    """Every date that has attendance, newest first."""
    return sorted({row["date"] for row in read_attendance(csv_file)}, reverse=True)
