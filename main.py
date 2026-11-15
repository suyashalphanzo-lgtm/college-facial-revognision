"""College face recognition system. Start it with:  python main.py"""

from pathlib import Path

from app.face_database import build_database, load_or_build_database
from app.face_engine import FaceEngine
from app.photo import recognize_photo
from app.webcam import add_person, recognize_live

MENU = """
========== College Face Recognition ==========
  1. Add a new person (take photos with the webcam)
  2. Register everyone in data/people (rebuild the database)
  3. Recognize faces in a photo
  4. Live webcam recognition + attendance
  5. Exit
"""


def main() -> None:
    print("Loading the face models (YuNet finds faces, our own SFace recognizes them)...")
    engine = FaceEngine()
    load_or_build_database(engine)  # registers everyone first if there is no database yet
    while True:
        print(MENU)
        choice = input("Choose 1-5: ").strip()
        if choice == "1":
            add_person(engine, input("Name of the person: ").strip())
        elif choice == "2":
            build_database(engine)
        elif choice == "3":
            # "Copy as path" in Windows Explorer wraps the path in quotes.
            photo_path = input("Path to the photo: ").strip().strip("\"'")
            recognize_photo(engine, Path(photo_path))
        elif choice == "4":
            recognize_live(engine)
        elif choice == "5":
            break
        else:
            print("Please type a number from 1 to 5.")


if __name__ == "__main__":
    main()
