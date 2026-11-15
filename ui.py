"""College face recognition system in a window. Start it with:  python ui.py

It does the same things as the text menu in main.py, with buttons instead of typed numbers.
"""

import ctypes
import math
import os
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from app import config
from app.attendance import AttendanceRegister
from app.cameras import camera_names
from app.face_database import build_database, count_photos, load_or_build_database
from app.face_engine import FaceEngine
from app.photo import label_and_save
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

VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
SECONDS_BEFORE_EACH_PHOTO = 3
PHOTO_FILE_TYPES = [("Photos", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]


class FaceRecognitionApp:
    def __init__(self, root: tk.Tk, engine: FaceEngine):
        self.root = root
        self.engine = engine
        self.database = load_or_build_database(engine)  # registers everyone first if there is no database yet
        self.attendance = AttendanceRegister()
        self.camera = None  # a CameraThread while the camera is on (or still opening)
        self.camera_number = chosen_camera()
        self.camera_labels = []  # "0: Integrated Webcam", ... as shown in the Camera list
        self.camera_start_time = 0.0
        self.camera_is_on = False
        self.mode = None  # "register" or "live" while the camera is on
        self.next_update = None
        self.person_name = ""
        self.photos_taken = 0
        self.next_photo_time = None

        self.build_window()
        self.find_cameras()
        self.refresh_lists()
        self.show_camera_off()
        self.message.configure(text="Add a person, or start the live camera.")

    def build_window(self) -> None:
        self.root.title("College Face Recognition (our own SFace)")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        video_area = tk.Frame(self.root, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, bg="black")
        video_area.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="n")
        video_area.pack_propagate(False)  # stay the same size whatever picture is shown
        self.video = tk.Label(video_area, bg="black", fg="white", font=("Segoe UI", 14))
        self.video.pack(fill="both", expand=True)
        # The Camera list sits right below the picture, using the height the button panel leaves spare.
        camera_row = ttk.Frame(self.root)
        camera_row.grid(row=1, column=0, padx=10, pady=(8, 0), sticky="w")
        ttk.Label(camera_row, text="Camera:").pack(side="left")
        self.camera_list = ttk.Combobox(camera_row, state="readonly", width=40)
        self.camera_list.pack(side="left", padx=(6, 0))
        self.camera_list.bind("<<ComboboxSelected>>", lambda event: self.choose_camera())
        ttk.Button(camera_row, text="Refresh", command=self.find_cameras).pack(side="left", padx=(6, 0))

        self.message = ttk.Label(self.root, font=("Segoe UI", 13, "bold"), wraplength=VIDEO_WIDTH)
        self.message.grid(row=2, column=0, padx=10, pady=10, sticky="nw")
        self.root.rowconfigure(2, weight=1)  # spare height goes below the message, not between video and message

        panel = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        panel.grid(row=0, column=1, rowspan=3, sticky="ns")

        add_box = ttk.LabelFrame(panel, text="Add a person", padding=8)
        add_box.pack(fill="x")
        ttk.Label(add_box, text="Name:").pack(anchor="w")
        self.name_entry = ttk.Entry(add_box, width=36)
        self.name_entry.pack(fill="x")
        self.name_entry.bind("<Return>", lambda event: self.start_registration())
        add_button(add_box, "Take 5 photos with the webcam", self.start_registration)
        add_button(add_box, "Open the photos folder", self.open_photos_folder)
        add_button(add_box, "Register photos from the folders", self.register_from_folders)

        recognize_box = ttk.LabelFrame(panel, text="Recognize", padding=8)
        recognize_box.pack(fill="x", pady=(8, 0))
        add_button(recognize_box, "Start live camera + attendance", self.start_live)
        add_button(recognize_box, "Recognize faces in a photo...", self.recognize_photo)
        add_button(recognize_box, "Stop camera", self.stop_camera)

        self.people_box = ttk.LabelFrame(panel, padding=8)
        self.people_box.pack(fill="both", expand=True, pady=(8, 0))
        self.people_list = tk.Listbox(self.people_box, height=4)
        self.people_list.pack(fill="both", expand=True)

        self.present_box = ttk.LabelFrame(panel, padding=8)
        self.present_box.pack(fill="both", expand=True, pady=(8, 0))
        self.present_list = tk.Listbox(self.present_box, height=4)
        self.present_list.pack(fill="both", expand=True)
        add_button(self.present_box, "Open the attendance file", self.open_attendance_file)

    def find_cameras(self) -> None:
        """Fill the Camera list with every camera Windows knows, by name, in OpenCV's order."""
        names = camera_names()
        if not names:  # another system, or Windows couldn't say: plain numbers to try
            names = [f"Camera {number}" for number in range(3)]
        self.camera_labels = [f"{number}: {name}" for number, name in enumerate(names)]
        self.camera_list.configure(values=self.camera_labels)
        if self.camera_number >= len(self.camera_labels):
            self.camera_number = 0  # the chosen camera isn't connected now (the saved choice is kept)
        self.camera_list.current(self.camera_number)

    def camera_label(self, camera_number: int) -> str:
        if camera_number < len(self.camera_labels):
            return self.camera_labels[camera_number]
        return str(camera_number)  # the list was refreshed and this camera is gone

    def choose_camera(self) -> None:
        self.camera_number = self.camera_list.current()
        remember_camera(self.camera_number)
        if self.camera is not None:  # the camera is on: carry on with the new one
            self.camera.stop()
            self.start_camera_thread()

    def refresh_lists(self) -> None:
        self.people_list.delete(0, "end")
        for name in sorted(self.database):
            photos = count_photos(config.PEOPLE_DIR / name)
            hint = ", add more" if photos < 3 else ""
            self.people_list.insert("end", f"{name}: {photos} photo(s){hint}")
        self.people_box.configure(text=f"Registered people ({len(self.database)})")

        self.present_list.delete(0, "end")
        for name in sorted(self.attendance.present_today):
            self.present_list.insert("end", name)
        self.present_box.configure(text=f"Present today ({len(self.attendance.present_today)})")

    def show_picture(self, image: np.ndarray) -> None:
        """Show an OpenCV picture in the video area, scaled to fit."""
        height, width = image.shape[:2]
        scale = min(VIDEO_WIDTH / width, VIDEO_HEIGHT / height)
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
        picture = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        self.video.configure(image=picture, text="")
        self.video.image = picture  # Tk only shows the picture while Python keeps a reference to it

    def show_camera_off(self) -> None:
        self.video.configure(image="", text="Camera is off")
        self.video.image = None

    def start_registration(self) -> None:
        name = self.name_entry.get().strip()
        if not is_valid_name(name):
            messagebox.showwarning(
                "Add a person", 'Type the person\'s name first. It can\'t use any of these: < > : " / \\ | ? *'
            )
            return
        self.stop_camera()
        self.person_name = name
        self.photos_taken = 0
        self.start_camera("register")

    def registration_step(self, frame: np.ndarray) -> None:
        """Take a photo when the countdown is over and exactly one face is visible."""
        faces = self.engine.detect_faces(frame)
        if self.next_photo_time is None:
            self.next_photo_time = time.monotonic() + SECONDS_BEFORE_EACH_PHOTO
        if time.monotonic() >= self.next_photo_time and len(faces) == 1:
            save_photo(config.PEOPLE_DIR / self.person_name, frame)
            self.photos_taken += 1
            if self.photos_taken == len(POSES):
                self.stop_camera()
                return
            self.next_photo_time = time.monotonic() + SECONDS_BEFORE_EACH_PHOTO

        seconds_left = math.ceil(self.next_photo_time - time.monotonic())
        if seconds_left > 0:
            status = f"photo in {seconds_left}..."
        elif not faces:
            status = "no face found"
        else:
            status = "only one person should be in view"
        pose = POSES[self.photos_taken]
        self.message.configure(text=f"Photo {self.photos_taken + 1} of {len(POSES)}: {pose} ({status})")

        for face in faces:
            x, y, width, height = (int(value) for value in face[:4])
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 180, 0), 2)
        self.show_picture(frame)

    def start_live(self) -> None:
        if not self.database:
            messagebox.showinfo("Live camera", "No one is registered yet. Add a person first.")
            return
        self.start_camera("live")

    def live_step(self, frame: np.ndarray) -> None:
        results = recognize_faces(self.engine, self.database, frame)
        newly_present = self.attendance.update([result.name for result in results if result.name != UNKNOWN])
        if newly_present:
            self.message.configure(text=f"Marked present at {datetime.now():%H:%M:%S}: {', '.join(newly_present)}")
            self.refresh_lists()
        draw_results(frame, results)
        self.show_picture(frame)

    def recognize_photo(self) -> None:
        if not self.database:
            messagebox.showinfo("Recognize a photo", "No one is registered yet. Add a person first.")
            return
        self.stop_camera()
        photo_path = filedialog.askopenfilename(title="Choose a photo", filetypes=PHOTO_FILE_TYPES)
        if not photo_path:
            return
        image = cv2.imread(photo_path)
        if image is None:
            messagebox.showerror("Recognize a photo", f"Could not open this file as a photo:\n{photo_path}")
            return

        results, result_path = label_and_save(self.engine, self.database, image, Path(photo_path))
        self.show_picture(image)
        known = [result.name for result in results if result.name != UNKNOWN]
        summary = f"Found {len(results)} face(s). Recognized: {', '.join(known) or 'nobody'}."
        self.message.configure(text=f"{summary}\nA labelled copy is saved in data/results/{result_path.name}")

    def register_from_folders(self) -> None:
        self.stop_camera()
        self.rebuild_database()
        self.message.configure(text=f"People registered: {len(self.database)}")

    def rebuild_database(self) -> None:
        self.message.configure(text="Registering everyone in the photos folder...")
        self.root.update_idletasks()  # show the message before the work starts
        self.database = build_database(self.engine)
        self.refresh_lists()

    def open_photos_folder(self) -> None:
        config.PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(config.PEOPLE_DIR)  # opens it in File Explorer (Windows only)

    def open_attendance_file(self) -> None:
        if config.ATTENDANCE_FILE.exists():
            os.startfile(config.ATTENDANCE_FILE)
        else:
            messagebox.showinfo("Attendance", "No one has been marked present yet.")

    def start_camera(self, mode: str) -> None:
        self.stop_camera()
        self.mode = mode
        self.start_camera_thread()
        self.update_camera()

    def start_camera_thread(self) -> None:
        self.camera = CameraThread(self.camera_number)  # opens it in the background; the window keeps working
        self.camera_start_time = time.monotonic()
        self.camera_is_on = False
        self.next_photo_time = None  # registration's countdown starts with the camera's first picture

    def update_camera(self) -> None:
        """Check on the camera thread and handle its newest picture, if there is one. Then ask Tk to
        call this again 10 ms later. In between, Tk redraws the window and reacts to the buttons."""
        state = self.camera.state
        if state == "could not open":
            self.stop_camera()
            self.message.configure(text="")
            messagebox.showerror("Webcam", CAMERA_HELP)
            return
        if state == "could not read":
            self.stop_camera()
            self.message.configure(text="Could not read a picture from the webcam.")
            return
        if state == "opening":
            seconds = time.monotonic() - self.camera_start_time
            self.message.configure(text=opening_message(self.camera_label(self.camera.camera_number), seconds))
        elif not self.camera_is_on:  # the camera has just turned on
            self.camera_is_on = True
            if self.mode == "live":
                self.message.configure(text=f"Looking for {len(self.database)} registered people...")

        frame = self.camera.take_newest_frame()
        if frame is not None:
            if self.mode == "register":
                self.registration_step(frame)
            else:
                self.live_step(frame)
        if self.camera is not None:
            self.next_update = self.root.after(10, self.update_camera)

    def stop_camera(self) -> None:
        """Turn the camera off. If photos were being taken, register the person with the photos taken so far."""
        if self.next_update is not None:
            self.root.after_cancel(self.next_update)
            self.next_update = None
        if self.camera is not None:
            if self.camera.state == "opening":
                self.message.configure(text="")  # clear "Opening the webcam..."
            self.camera.stop()  # its thread turns the webcam off; no need to wait for it here
            self.camera = None
        registration_ended = self.mode == "register" and self.photos_taken > 0
        self.mode = None
        self.show_camera_off()
        if registration_ended:
            self.rebuild_database()
            self.message.configure(text=f"{self.person_name} is registered with {self.photos_taken} photo(s).")

    def close(self) -> None:
        camera = self.camera
        self.stop_camera()
        if camera is not None:
            camera.stop(wait_seconds=3)  # let its thread turn the webcam off before Python exits
        self.root.destroy()


def add_button(parent: tk.Widget, text: str, command) -> None:
    ttk.Button(parent, text=text, command=command).pack(fill="x", pady=(4, 0))


def main() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # keeps text sharp on screens with display scaling
    except (AttributeError, OSError):
        pass  # not Windows, or a version without display scaling
    print("Loading the face models (YuNet finds faces, our own SFace recognizes them)...")
    engine = FaceEngine()
    root = tk.Tk()
    FaceRecognitionApp(root, engine)
    root.mainloop()


if __name__ == "__main__":
    main()
