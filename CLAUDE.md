# Facial Recognition Learning Project

## Purpose

The purpose of this project is to learn how a facial recognition system is built, step by step. The system is meant for college use: register people's faces, recognize them in photos or live on a webcam, and keep an attendance register.

## Learning philosophy

Implement one small piece at a time. Understand each piece before moving on to the next.

Decision: at the user's request, the complete basic system was built in one go, to be studied afterwards. To keep the "one piece at a time" spirit, study it file by file in the order given under "Study order".

## About this file

This file is the project's living technical textbook. Every important technical decision, assumption, implementation detail, and lesson should be documented here. Only document what actually exists and decisions that have actually been made.

## Current status

**Complete basic system, with measured accuracy and a window.** All four steps of face recognition work end to end: add a person with the webcam, register everyone from photo folders, recognize faces in a photo, and live webcam recognition with attendance. They can be used from a window with buttons (`python ui.py`), which lists the computer's cameras by name to choose from, or from a text menu (`python main.py`). Recognition accuracy was measured on the LFW face dataset, and two improvements came out of it: a stricter match threshold (0.46 instead of 0.363), so strangers no longer get registered people's names, and guided registration that takes one photo for each of five head poses. On a friend's laptop the window froze ("Not Responding") while it opened the webcam, so the webcam now has a thread of its own and is opened with DirectShow first (see "Lessons from the first run on another laptop"). Since then it has run on this PC's real camera (a phone used as a webcam through Iriun), from a fresh download: registration took its five photos, and live recognition started.

**There is also a web version** (`python web_app.py`: Flask, with the page in HTML, CSS and JavaScript; see "The web version"). It does everything the window does, on one page in the browser: live attendance, registration with the camera or from uploaded photos, recognizing a photo, attendance for any date with CSV downloads, and deleting a person. On a computer screen it all fits in the window, so nothing needs scrolling. It was tested with the real models, a pretend camera and a real browser, but not yet with a real camera.

**The app recognizes faces only with our own SFace** (see "Our own SFace"). The lab requires the project's own model, so OpenCV's pretrained SFace is not part of the app: our model is the only recognizer, run by OpenCV's general deep-learning engine, and faces are aligned by our own code. Only the face detector, YuNet, is still OpenCV's pretrained model. Two models were trained on Kaggle's free GPU (this laptop's GPU overheats). The app uses the second one, which trained three times longer (90 epochs in 10.7 hours) on training photos with webcam-style changes. Measured against OpenCV's SFace as a yardstick, it is a little less accurate: 99.13% against 99.35% on the standard LFW test, and 96.3% of registered people named against 99.4% in the college test. The longer training helped on the standard test (the first model scored 98.82%) but not in the college test: measured over 20 random draws, both of our models name about 97% of registered people.

History: Part 1A was the initial skeleton (folders and a `main.py` that printed one line). Then the complete basic system was built in one go. Then accuracy was measured on LFW and improved (see "Accuracy measured on the LFW dataset"). Then a basic window was added (`ui.py`). Then our own SFace was trained on Kaggle, tested against OpenCV's, and added to the app as a second recognizer behind a switch. Then a second model was trained for longer, with webcam-style changes to the training photos, and replaced the first. Then the project was put on GitHub for a friend, whose laptop showed that the window froze while opening the webcam, and the webcam got a thread of its own. Then OpenCV's SFace was taken out of the app, because the lab requires its own model: our SFace became the only recognizer, and alignment was written in our own code. Then the window got a Camera list, to choose between the cameras by name. Then a web version was built (Flask with HTML, CSS and JavaScript): first recognizing a photo, then, in one go, the live camera, registration and attendance history. Then its page was redesigned to fit in one window, because every feature needed scrolling on this computer.

## How to run

- Needs Python 3.10 or newer and `pip install -r requirements.txt` (opencv-python 4.8 or newer, numpy, pillow, pytest, flask). This computer has Python 3.13, OpenCV 5.0, Tk 8.6 and Flask 3.1.
- `python ui.py` opens the window. `python main.py` opens the text menu instead. If OpenCV's YuNet face detector is missing from `models/`, either one downloads it first (0.2 MB). Our own model, `models/our_sface.onnx`, comes with the project.
- `python web_app.py` starts the web version and opens `http://127.0.0.1:5000` in the browser. Keep its terminal open; Ctrl+C there stops it. Only one copy can run at a time. After changing its Python code or its HTML, stop it and start it again; after changing its CSS or JavaScript, reloading the page is enough.
- `python -m pytest` runs the tests. Use `python -m pytest`, not plain `pytest`, so Python can find the `app` package.

Typical use of the window (`python ui.py`). First choose the camera in the Camera list under the picture; the app remembers the choice.

1. Register each person: type the name and click "Take 5 photos with the webcam". The text under the picture says which pose to make (look straight, turn slightly left, turn slightly right, tilt up, tilt down) and counts down 3 seconds before each photo. It finishes by itself after the fifth. Or click "Open the photos folder", make a folder named after each person with 3–5 photos of them, and click "Register photos from the folders".
2. "Start live camera + attendance": names appear on the picture, and people are added to "Present today" (and `data/attendance.csv`). "Stop camera" turns it off.
3. "Recognize faces in a photo...": choose a photo; the labelled result is shown, and a full-size copy is saved in `data/results/`.

The text menu (`python main.py`) does the same with typed numbers: option 1 registers (press SPACE for each of the five photos in the webcam window), option 2 registers from the folders, option 3 recognizes a photo (type its path), option 4 is live recognition (press Q to stop).

Settings such as the match threshold are in `app/config.py`. The camera is chosen in the window, and the text menu uses the same choice.

Training our own SFace needs `pip install -r training/requirements.txt` (adds PyTorch and onnx). All its steps run from the project folder:

- `python -m training.datasets`: packs the training photos and aligns LFW (needs the files in `datasets/`, see "Our own SFace").
- `python -m training.train`: trains. Too hot for this laptop; use the Kaggle notebook instead (see "Training on Kaggle").
- `python -m training.export_onnx`: turns the trained network into `models/our_sface.onnx`. It takes the second model (`training/runs/mobilefacenet_sface_v2/`); add `--run mobilefacenet_sface` for the first one.
- `python -m training.evaluate`: tests our model on LFW (about a minute), with the mirror trick as the app uses it. `--compare-opencv` also tests OpenCV's pretrained SFace the same way, as a yardstick (it downloads it, 37 MB; the app never uses it). `--mirror on` or `--mirror off` tests with or without the mirror trick.
- `python -m training.make_kaggle_notebook`: rebuilds `training/our_sface_kaggle.ipynb` after the training code changes.

## How face recognition works

A computer can't compare two faces directly, so every face is turned into a list of numbers and the numbers are compared. There are four steps.

1. **Detection: where are the faces?** OpenCV's pretrained YuNet model scans the image and returns, for every face, a box, five landmark points (both eyes, nose tip, both mouth corners) and a confidence score from 0 to 1. Detection only finds faces; it doesn't know whose they are.
2. **Embedding: what does this face look like, as numbers?** First the face is *aligned* by our own code (`align_face`): it is turned, scaled and moved so that its five landmarks land as close as possible on a standard template, which puts the eyes level, and cut out at 112×112 pixels. Then our own SFace model (see "Our own SFace") turns it into an *embedding*, a list of 128 numbers. It was trained so that photos of the same person give similar embeddings and photos of different people give different ones. Each embedding is scaled to length 1. The mirror image of the aligned face is embedded too, and the two embeddings are averaged (the "mirror trick").
3. **Registration: remember the known people.** For each person, the embeddings of all their photos are averaged into one embedding (scaled back to length 1) and saved under their name in `data/face_database_ours.json`.
4. **Matching: who is this?** A new face's embedding is compared with every registered person using *cosine similarity*. For embeddings of length 1 this is simply the dot product: close to 1 means the same face, close to 0 means unrelated faces. The most similar person wins, but only if the similarity reaches `MATCH_THRESHOLD` (0.48); otherwise the face is "Unknown".

Attendance sits on top of this: a recognized person is marked present once per day, after being recognized in `FRAMES_TO_CONFIRM` (10) video frames in a row.

## Study order

| Order | File | What it covers |
| --- | --- | --- |
| 1 | `app/config.py` | Every setting and file location |
| 2 | `app/face_engine.py` | Steps 1 and 2: loading the two models, detection, our own alignment code, embeddings with our model |
| 3 | `app/face_database.py` | Step 3: registration, averaging, saving and loading the JSON database, building it when there is none yet |
| 4 | `app/recognizer.py` | Step 4: cosine similarity, the threshold, drawing boxes and names |
| 5 | `app/attendance.py` | The attendance rules and the CSV file |
| 6 | `app/photo.py` | Menu option 3: all the steps on one photo |
| 7 | `app/webcam.py` | Menu options 1 and 4: guided registration, and the same steps on every webcam frame. Opening the webcam, remembering the chosen camera, and the camera thread the window uses. (`app/cameras.py`, which asks Windows for the cameras' names, is low-level and can be skipped.) |
| 8 | `main.py` | The text menu that ties everything together |
| 9 | `ui.py` | The window: the same features behind buttons, with the camera picture shown inside the window |
| 10 | `web_app.py`, `app/camera_session.py`, `web/templates/index.html`, `web/static/style.css`, `web/static/app.js` | The web version: Flask's routes, the camera session the server keeps running, and the page's HTML, CSS and JavaScript |
| 11 | `tests/` | Checks for matching, the database, attendance, name checking, the face alignment, the camera thread and every route of the web version (with a pretend camera), without a camera or models; `test_our_sface.py` checks the network and the SFace loss |
| 12 | `training/mobilefacenet.py` | Our own SFace: the network that turns a face into 128 numbers |
| 13 | `training/sface_loss.py` | The SFace loss: pulling each face towards its own person, pushing it away from the others |
| 14 | `training/datasets.py` | Packing the training photos; aligning LFW with the app's own code |
| 15 | `training/train.py` | The training loop: batches, learning rate, 16-bit arithmetic, saving progress |
| 16 | `training/evaluate.py` | The two LFW tests, how the threshold is chosen, and the comparison with OpenCV's SFace |
| 17 | `training/export_onnx.py` | Saving the network as an ONNX file OpenCV can load |
| 18 | `training/make_kaggle_notebook.py` | Building the Kaggle notebook from the project's own files |

## Technical decisions

- **Built on OpenCV, not the `face_recognition`/dlib library.** OpenCV was already installed and needs no C++ compiler (dlib usually has to be compiled on Windows). Its pretrained YuNet model finds faces, and its deep-learning engine (`cv2.dnn`) runs our own SFace. The project first used OpenCV's pretrained SFace too, until our own replaced it (see "Only our own SFace in the app").
- **YuNet comes from the official OpenCV model zoo** (`opencv/opencv_zoo` on GitHub): `face_detection_yunet_2023mar.onnx` (227 KB). This version needs OpenCV 4.8 or newer, hence `opencv-python>=4.8`. OpenCV's SFace (`face_recognition_sface_2021dec.onnx`, 37 MB) comes from there too, but only `python -m training.evaluate --compare-opencv` downloads it.
- **YuNet downloads automatically** into `models/` the first time it is needed. The download is saved under a temporary name and only renamed when complete, so a broken download is never loaded.
- **One averaged embedding per person** rather than one per photo: the database stays small and matching needs one comparison per person. Matching against each photo separately was measured and was no better (see LFW lessons).
- **The database is a JSON file** so it can be opened and read: each name maps to its 128 numbers, rounded to 6 decimals.
- **The folders in `data/people/` are the source of truth.** The database is always rebuilt from them (menu option 2; option 1 rebuilds automatically after taking photos). To remove someone, delete their folder and rebuild.
- **Registration uses the biggest face in each photo**, in case someone else is in the background. Photos without a face, or that can't be read, are skipped with a message. Accepted file types: jpg, jpeg, png, bmp, webp. A tip is printed for anyone with fewer than 3 usable photos, because people registered with 1 photo were recognized less often (with our model, about 77% against 95–97% with 3–5 photos).
- **Match threshold 0.48**: the lowest threshold at which at most 0.4% of strangers get a name with 100 people registered, measured for our model (see "Our own SFace"). That rule came from OpenCV's SFace, which the app used first. Its threshold started at 0.363, the value in OpenCV's SFace example, but measuring on LFW showed that value gave about 15% of strangers a registered person's name when 100 people were registered. 0.46 brought that down to 0.4% while registered people were still named 99.4% of the time. See "Accuracy measured on the LFW dataset".
- **The mirror trick** (`MIRROR_TRICK` in `app/config.py`, on). `get_embedding` also embeds the mirror image of the aligned face (`cv2.flip`) and averages the two embeddings. Measured on LFW, it helped our first model a little. For our second model it makes no measurable difference with 3–5 photos per person and helps a little with 1, so it stays on (see "Our models against the target"). For OpenCV's SFace it changed nothing. It doubles the time per face, about 3 ms more with our small model. It changes every embedding, so after changing it everyone must be registered again.
- **Only our own SFace in the app.** The lab requires the project's own model, so the app has no other recognizer and never downloads OpenCV's SFace. (Before, a `RECOGNIZER` switch in `app/config.py` chose between OpenCV's SFace, the default because it was more accurate, and ours.) If `models/our_sface.onnx` is missing, the app stops with a message saying how to get it back.
- **The database file is `data/face_database_ours.json`.** One model's numbers can't be compared with another's, so OpenCV's SFace had its own file, `data/face_database.json`. That old file may still be on this computer; the app ignores it. When there is no database file yet but there are people folders, `load_or_build_database` registers everyone first.
- **Our own alignment code** (`align_face` and `similarity_transform` in `app/face_engine.py`) instead of OpenCV's `alignCrop`. alignCrop belongs to OpenCV's `FaceRecognizerSF` class, which can only be created with a model file, but the LFW test photos must be aligned before our model exists (on Kaggle, before training). So the same calculation is written out: Umeyama's least-squares fit of a turn, a scale and a shift (no stretching, never a mirror image) that moves the 5 landmarks onto the standard template. Then `cv2.warpAffine` cuts out the 112×112 face. On all 13,233 LFW photos it gave the same crops as alignCrop to within 1 brightness step, in 0.03% of the pixels: OpenCV does part of the calculation with 32-bit numbers and rounds the template's centre. The embeddings of the two kinds of crops had a similarity of 0.999997 or more, and both LFW tests gave the same results (only "named with 1 photo" moved, from 74.9% to 75.1%).
- **Our model is run by OpenCV's deep-learning engine** (`cv2.dnn`; `load_network` and `run_network` in `app/face_engine.py`), not through OpenCV's `FaceRecognizerSF` class. The only preparation is turning OpenCV's BGR colour order into RGB, which is what `FaceRecognizerSF` did: on 2,000 LFW faces the two gave exactly the same numbers. `training/evaluate.py` and `training/export_onnx.py` run models the same way.
- **Detection runs on a copy shrunk to at most 640 pixels** (`DETECTION_MAX_SIZE`). The boxes and landmarks are then scaled back up, so alignment and embedding use the full-quality image. See "Lessons from the first tests" for why.
- **Detection confidence 0.8** (`DETECTION_CONFIDENCE`): anything the detector is less sure about is ignored.
- **The webcam picture is mirrored**, so it moves like a mirror on screen. Photos taken with option 1 are saved mirrored as well, so registration and live recognition see faces the same way.
- **Option 1 is guided: one photo for each of five head poses** (`POSES` in `app/webcam.py`), so the person is registered from several angles. A photo is only saved when exactly one face is visible, so nobody gets registered under someone else's name. It stops by itself after the fifth photo; Q stops early and keeps the photos taken so far.
- **Attendance rules:** at most once per person per day, and only after 10 frames in a row, so a single wrong guess can't mark someone present. The CSV has the columns `name,date,time` and is re-read at start-up, so restarting doesn't mark anyone twice.
- **An OpenCV 5 log message is hidden while the models load.** OpenCV 5 prints "Targets are not supported by the new graph engine for now" when these models are created; it is harmless.
- **An interactive menu instead of command-line arguments**, because it is the easiest to use and to demo.
- **The window uses Tkinter**, which comes with Python, plus **Pillow** to turn OpenCV pictures into pictures Tkinter can show. The window and the text menu use the same `app/` modules; shared pieces are `is_valid_name` and `save_photo` in `app/webcam.py` and `label_and_save` in `app/photo.py`.
- **In the window, the webcam has a thread of its own** (`CameraThread` in `app/webcam.py`). Opening a webcam can take seconds, and every read waits for the camera's next picture. A window that waits can't redraw itself or react to its buttons, and Windows marks it "Not Responding". So the camera thread does all the waiting: it opens, reads and releases the camera, and keeps only the newest picture. The window never touches the camera, and the camera thread never touches the window (Tkinter only works from the thread that created it). Instead of a `while True` loop, `update_camera` checks on the camera thread, handles its newest picture and asks Tkinter (`root.after`) to call it again 10 ms later, so buttons keep working while the camera runs.
- **Only one camera thread uses the webcam at a time** (the lock `CameraThread.webcam_in_use`). Stop camera doesn't wait for the old thread to release the webcam, so a camera started right afterwards waits for that in its own thread. Closing the window waits up to 3 seconds for it, so the webcam is released properly before Python exits.
- **While the webcam opens, the message counts the seconds**, and after 20 seconds it says what to check. If the webcam can't be opened at all, a message box lists the usual causes (`CAMERA_HELP`): another app using it, Windows' camera privacy setting, the laptop's camera switch or key, and `CAMERA_INDEX`.
- **On Windows, the webcam is opened with DirectShow first, then Media Foundation** (`CAMERA_BACKENDS` in `app/webcam.py`). OpenCV's own first choice is Media Foundation, which works with some newer laptop cameras that DirectShow can't use. But on some computers it is very slow to open a camera, because it first loads the graphics card's video converters. DirectShow opens most webcams in about a second. For when Media Foundation is used, the converters are turned off (`OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0`; OpenCV reads it every time it opens a camera). A backend only counts once it gives a picture within 5 seconds (`SECONDS_TO_WAIT_FOR_FIRST_PICTURE`); otherwise the next one is tried. The text menu opens the webcam the same way.
- **The window's Camera list shows every camera by name** (`camera_names` in `app/cameras.py`). OpenCV opens cameras by number but can't tell their names, so the names are read from DirectShow through Windows' COM system, with ctypes (no extra library). DirectShow lists the cameras in the same order OpenCV's DirectShow code numbers them, so the numbers in the list are the ones OpenCV opens, and nothing is turned on to read them. It runs in a short-lived thread of its own, so its COM setup can't disturb the window's thread. If no names can be read (another system), the list offers Camera 0, 1 and 2. **Refresh** reads the list again, for a camera plugged in later.
- **The chosen camera is remembered** in `data/camera.txt` (`chosen_camera` and `remember_camera` in `app/webcam.py`), and the text menu uses it too. Until a camera is chosen, `CAMERA_INDEX` in `app/config.py` (0) is used. If the saved camera isn't connected, the list shows camera 0 but keeps the saved choice, so that camera is used again once it is plugged back in. Choosing another camera while the camera is on carries on with the same job: live recognition stays live, and registration keeps the photos taken so far and starts the countdown again.
- **The Camera list sits right below the picture**, where the button panel leaves spare height. In a box of its own at the top of the panel it made the window 68 pixels taller (563 to 631), which could push the bottom buttons off a small screen with display scaling; below the picture it adds 7.
- **Registration in the window is hands-free:** a 3-second countdown (`SECONDS_BEFORE_EACH_PHOTO`) before each of the five poses, and a photo is only taken when exactly one face is visible. The first countdown starts with the camera's first picture, so a slow webcam doesn't make the first photo be taken the moment it opens. If registration is stopped early (Stop camera, another button, or closing the window), the person is registered with the photos taken so far. The text menu still uses SPACE.
- **The window's picture area is 640×480**, the usual webcam size. Photos are scaled to fit it; the full-size labelled copy is in `data/results/`.
- **The registered-people list shows each person's number of photos** and says "add more" below 3, the same rule as the tip printed during registration.
- **The window asks Windows not to blur it on screens with display scaling** (`SetProcessDpiAwareness`), the same thing Python's own IDLE editor does.
- **"Open the photos folder" and "Open the attendance file"** use `os.startfile`, which opens them like a double-click in File Explorer would (Windows only).
- **`.gitignore`** keeps personal data (`data/people/`, both database files, attendance, results) out of git, along with OpenCV's downloaded models, the datasets and the saved training networks. It lets in our own model (`models/our_sface.onnx`, 4 MB) and the two training logs (`training/runs/*/log.csv`), so someone who downloads the project can use our model without training it.
- **The project is on GitHub** as the private repository `Aaditya13050/college-face-recognition`, to share with the friend writing the lab file. `README.md` is its front page (setup and how to run), `HOW_TO_RUN.md` the short version for someone who only wants to run it, and `BUILD_LOG.md` tells the story of how our own SFace was built, in order, for readers new to the project.

## Lessons from the first tests

These tests were made with OpenCV's SFace, which the app used then, on public-domain official portraits of US politicians from Wikimedia Commons, deleted afterwards. The webcam screens were tested with a simulated camera that fed photos through the real webcam code, without opening a camera or any windows.

- **Similarity values seen with SFace.** Same person, different photos: Obama's 2009 and 2012 portraits scored 0.82; Biden's 2013 and 2021 portraits (8 years apart) scored 0.69. Different people never scored above 0.26. Comparing two photos at a time like this, 0.363 looked safe, but the LFW measurement showed it is too lenient once a whole class is registered.
- **Recognition on photos that were not used for registration:** Obama 0.77 and Biden 0.70, both named correctly. Two people who were not registered scored 0.17 and 0.26 and were shown as "Unknown".
- **Group photo with 12 people:** all 12 faces were detected. Obama, looking sideways, scored 0.47 and was recognized. Biden, turned to the side and registered with a single portrait taken 10 years later, scored 0.34 and stayed "Unknown". With a second registration photo he scored 0.43, enough at the old threshold of 0.363 but not at the current 0.46. **Lesson: register several photos per person, from different angles.**
- **The detector misses faces that are too big.** A 3356×2687 close-up portrait gave 0 faces at full size; shrunk, the same face was found with 95% confidence. When images were shrunk to 1280 pixels, faces about 900 pixels tall were still missed and a 700-pixel face was only just found. At 640 pixels every close-up was found, and the group photo still gave all 12 faces. That is why `DETECTION_MAX_SIZE` is 640. Raise it to 1280 only for big group photos where every face is tiny.
- **Mirrored webcam-style frames were still recognized.** The person was registered with an unmirrored portrait and recognized in mirrored frames.
- **Our dot product equals OpenCV's own cosine similarity.** `FaceRecognizerSF.match(..., FR_COSINE)` gave the same 0.820 for the two Obama portraits.
- **Speed on this computer's CPU:** detection takes about 20–30 ms at 640 pixels, and each face's embedding about 15–50 ms. The very first embedding after start-up took about 3 seconds while the model warmed up. Measured again later in a fresh process, the first detection took 10 ms and the first embedding 23 ms, so those 3 seconds don't happen every time.

## Accuracy measured on the LFW dataset

LFW ("Labeled Faces in the Wild") is a standard public dataset of 13,233 news photos of 5,749 people. It was downloaded for this measurement and deleted afterwards. The measurement was made with OpenCV's SFace, which the app used then; the same tests for our own model are in "Our models against the target". It copied a college class:

- 100 people with at least 6 photos in LFW were registered with up to 5 photos each (and, for the smaller class, the first 30 of them).
- Up to 5 other photos of each registered person were test photos, which should get the right name.
- One photo each of 1,000 other people were the strangers, who should be "Unknown".
- The comparison of methods used one random draw of people; the table below is the average of two other draws.

Results for the method the app uses (average of each person's photos):

| Registered people | Photos per person | Threshold | Registered people named correctly | Strangers given a name |
| --- | --- | --- | --- | --- |
| 100 | 5 | 0.363 | 99.6% | 15.1% |
| 100 | 5 | 0.44 | 99.4% | 0.9% |
| 100 | 5 | **0.46** | 99.4% | 0.4% |
| 100 | 5 | 0.48 | 98.9% | 0.1% |
| 100 | 3 | 0.46 | 99.3% | 0.4% |
| 100 | 1 | 0.46 | 95.9% | 0.1% |
| 30 | 5 | 0.363 | 100% | 5.9% |
| 30 | 5 | 0.46 | 100% | 0.1% |
| 30 | 1 | 0.46 | 95.5% | 0.0% |

What this taught:

- **The real problem was strangers getting names, not registered people being missed.** OpenCV's 0.363 was chosen for comparing two photos (one-to-one). Here a stranger is compared with every registered person and the best match counts, so they get many chances to reach a high similarity by luck.
- **The more people are registered, the more strangers get named** at the same threshold: 5.9% with 30 people and 15.1% with 100, at 0.363. One-to-many recognition needs a stricter threshold than one-to-one.
- **0.46 was chosen** because registered people (with 3–5 photos) were named as often as at lower thresholds, while strangers dropped to 0.4% with 100 people and 0.1% with 30. At 0.48, registered people start being missed.
- **More photos per person help:** at 0.46, 1 photo gave about 96% correct, 3–5 photos 99.3–100%.
- **Naming a registered person as a different registered person was rare:** at most 0.3%.
- **Two ideas were tried and not used.** Both were compared with 100 people and 5 photos each, with the threshold set so that 1% of strangers got a name:
  - *Mirror trick* (also embedding the mirror image of the face and combining the two): 99.7% instead of 99.4% correct. That is a single test photo out of 351, and it doubles the time per face (13 ms to 27 ms). Measured again later with `training/evaluate.py`, it made no difference for OpenCV's SFace either, but it helped our first model a little (see "Our models against the target").
  - *Closest photo* (comparing with each registered photo and keeping the best, instead of the average): the same accuracy, but at 0.363 it gave even more strangers a name (23% instead of 13%).
- **Small faces still work:** with test photos shrunk until the face was 34 pixels wide, 99.7% were still named; at 24 pixels, 97.7% were (4 faces were no longer found). This was measured with the closest-photo and mirror setup at 0.363.

## Lessons from building the window

The window was tested in two ways, both with a simulated camera: its code was driven with the window hidden and dialogs replaced by recorders (13 checks covering every button), and it was shown on screen for a few seconds to take screenshots.

- **Tkinter only shows a picture while Python keeps a reference to it.** That is why `show_picture` stores it in `self.video.image`; without that line the picture area would stay empty.
- **A widget that spans several grid rows makes Tkinter spread the spare height over those rows.** The button panel is taller than the video, and the first screenshot showed a gap between the video and the message. `rowconfigure(1, weight=1)` sends the spare height below the message instead.
- **Long lines get cut off in a list box.** The first screenshot showed "add more photos" cut off, so the entries became shorter ("Joe Biden: 2 photo(s), add more") and the panel a little wider.
- **Windows' list of devices doesn't show every camera.** On this computer, DirectShow lists "Iriun Webcam" (a phone used as a webcam) and "OBS Virtual Camera", but Windows' list of devices (`Win32_PnPEntity`, what Device Manager shows) has only Iriun: OBS's virtual camera is a program registered with DirectShow, not a device. OpenCV can open both, so the Camera list takes its names from DirectShow.

## Lessons from the first run on another laptop

A friend downloaded the project from GitHub, installed it, typed a name and clicked "Take 5 photos with the webcam".

- **The window froze while opening the webcam.** The screenshot showed "(Not Responding)" in the title and "Camera is off" in the picture area. The new message "Opening the webcam..." was drawn over the end of the old one ("...he live camera."), so the window had stopped redrawing right after the program asked for the camera. The terminal showed "Opening the webcam..." and nothing after it, so Python was still waiting inside OpenCV for the camera. Windows marks a window "Not Responding" when it hasn't reacted for about 5 seconds. The simulated camera of the earlier tests opened instantly, so they couldn't show this.
- **Why opening was so slow on that laptop isn't known for sure.** The best-known cause is Media Foundation, OpenCV's first choice on Windows, loading the graphics card's video converters when it opens a camera; OpenCV has the `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS` setting to turn that off. Another app holding the camera could also do it. The fix works whatever the cause: the camera has its own thread, so the window keeps working however long the camera takes; DirectShow is tried first; and Media Foundation's converters are turned off.
- **What OpenCV's camera code does, read from its source:** DirectShow opens a camera at 640×480 unless asked otherwise, and each read waits up to 1 second for a new picture. Media Foundation waits up to 10 seconds. Both wait for a new picture instead of returning the last one again, so the camera thread never spins in a busy loop.
- **A thread was tested with a pretend camera that is slow on purpose.** A fake camera that took 4 seconds to open, driving the real window with the window hidden, showed four things. The window kept working: no round of its event loop took more than a few hundred milliseconds, the same as with no camera at all. The first countdown started after the camera opened. Stopping the camera, or closing the window, while it was still opening released it. A camera that couldn't be opened gave the message box. The tests in `tests/test_webcam.py` check the camera thread the same way, in under a second. The one-camera-at-a-time test was checked by removing the lock: it failed, as it should.
- **An error inside a thread doesn't reach the window by itself.** If OpenCV raised an error in the camera thread, the thread would end quietly and the window would show "Opening the webcam..." forever. So `CameraThread.run` always sets a final state ("could not open" or "could not read"), and the camera is released even after an error.

## The web version

**Goal:** a more modern screen, and learning web development along the way. Flask with HTML and CSS was chosen over Streamlit (the quickest to learn, about 10–15 hours), CustomTkinter and PySide6, because the student already knows some HTML and CSS. With Flask the page is designed in plain HTML and CSS, and the skills carry over to any web project. Estimated learning time from zero, with that HTML and CSS: 25–35 hours.

**Status: complete.** It was planned in four steps. Step 1 was built on its own, and at the user's request steps 2–4 were then built in one go:

1. Recognize the faces in an uploaded photo, and list the registered people.
2. The live camera in the page: the camera choice, Start and Stop, names on the picture, and today's attendance updating by itself.
3. Registration with the camera (the 5 poses, a countdown and progress dots) or from uploaded photos.
4. Attendance for any date, CSV downloads, and deleting a person.

The window (`ui.py`) still works too, and both use the same `app/` modules.

**How it works:**

- **Three parts, in three languages.** `web_app.py` is the server, in Python with Flask: each route is a function that answers one address, and the list of routes is at the top of the file. `web/templates/index.html` is the page's structure (HTML), `web/static/style.css` its look (CSS), and `web/static/app.js` its behaviour (JavaScript, run by the browser). The page is sent once. After that, the JavaScript asks the server for data (JSON) and changes the page itself, so the page never reloads.
- **One window, no scrolling.** On a computer screen the page is exactly as tall as the window (`height: 100vh`), and a CSS grid shares it out: the camera on the left, the tools in the middle (Add person, Recognize and History, as tabs), and Present today and Registered people on the right. A list longer than its pane scrolls inside the pane. In a window narrower than 1000 pixels or shorter than 520 (a phone, or a small window), the panes are stacked instead and the page scrolls. The rules for computer screens are at the end of `style.css`.
- **The server keeps the camera going** (`CameraSession` in `app/camera_session.py`). A page can't run a loop on the server, so the session runs it: a worker thread takes the camera thread's newest picture, names the faces (live) or takes the registration photos, marks attendance, and keeps the newest labelled picture as a JPEG. The session also remembers everything the page shows: whether the camera is on, the registration's progress, the message.
- **The page asks for the status a few times a second** (`/api/status`: every 0.3 seconds while the camera is on, every second while it is off) and shows it: the camera list, the badge (Live, Registering, Opening), the countdown and pose over the picture, the five progress dots, today's arrivals and the registered people.
- **The camera picture is fetched one picture at a time** (`/api/frame.jpg`). The page asks for the newest picture, waits until it has arrived, and asks again. The server holds each request until there is a picture newer than the last one it sent (for up to 1 second). This was chosen over an endless MJPEG stream: every picture is an ordinary request, which is easier to follow, works with Flask's test client, and lets a slow computer go at its own pace instead of piling up pictures.
- **Registration works like the window's:** a 3-second countdown before each of the 5 poses, starting with the camera's first picture, and a photo only when exactly one face is visible. After the fifth photo (or Stop camera), everyone is registered again from the photo folders.
- **Lists are only redrawn when what they show has changed**, so a × button isn't replaced while it is being clicked.

**Technical decisions:**

- **Only this computer can open the site:** the server listens on 127.0.0.1, port 5000.
- **Only this computer's names are accepted as the request's host** (`127.0.0.1`, `localhost`). Otherwise a website could use a domain name that points at 127.0.0.1 to read this site ("DNS rebinding").
- **Changes are only accepted from the page itself.** Every POST must carry the header `X-Face-App: 1`, which `app.js` adds. Another website open in the same browser can't add it (the browser would first ask this server for permission, and it doesn't give it), so it can't press this site's buttons, like starting the camera or deleting a person. A request without the header gets 403.
- **Flask's debug mode stays off:** its error page can run Python code typed into a web page.
- **Only one copy can run.** Before loading the models, `main()` checks whether something already answers at 127.0.0.1:5000 (`already_running`). If so, it stops with a message saying how to stop the other copy. Without this check, a second copy started without an error, and the browser kept talking to the first (see the lessons). The check tries to connect for at most half a second: a running server answers in under a millisecond, while on Windows a port where nothing runs takes 2 seconds to refuse.
- **Answers are JSON.** A refusal the user should see (a name that isn't allowed, no one registered yet) is raised as `Refused` in `CameraSession` and sent as `{"error": "..."}` with status 400. The page shows the message next to what was clicked.
- **Names are put on the page with `textContent`, never `innerHTML`**, so a person's name is always shown as text and can't add HTML or JavaScript to the page.
- **Uploads are at most 50 MB together**, enough for 5 phone photos. The saved copy of a recognized photo gets its file name through `secure_filename`.
- **Deleting a person checks where the folder really is** (`resolve()`): it must be directly inside `data/people/`, so no name, like `..`, can delete anything else. Names Windows would change (ending with a dot or a space, `.` and `..`) are now refused everywhere, in the window too (`is_valid_name`).
- **Uploaded photos without a face are not kept**, because a person who isn't registered doesn't appear in the list where they could be deleted.
- **`save_photo` never overwrites a photo:** several uploaded photos can be saved within one tick of the clock, so a number is added when a file name is already taken.
- **One face job at a time:** the worker, the photo page and re-registering all take the `face_work` lock before using the models.
- **`create_app` takes the models, the database and the camera list as arguments**, so `tests/test_web_app.py` can test every route with stand-ins and a pretend camera, through Flask's test client.
- **Shared pieces moved into `app/`**, so the window and the web version use the same code: `count_photos` (`app/face_database.py`), `opening_message` (`app/webcam.py`), and reading attendance by date (`read_attendance`, `attendance_on` and `attendance_dates` in `app/attendance.py`).
- **The camera's column is as wide as the biggest 4:3 picture the window's height allows.** `--picture-height` in `style.css` is the window's height minus everything above and below the picture (the top bar, the pane's padding, the buttons, the gaps between the pane's parts, and the message line), and the column is 4/3 of it, plus the pane's padding and its 1-pixel line. The other two columns share the rest of the width. Inside the pane, the picture area takes the height that is left, and the picture fits it with container query units (`100cqw` and `100cqh`, the width and height of the element it sits in). So the picture is always 4:3 and as big as the window allows, and in a shorter window it gets smaller instead of pushing the page down.
- **A photo is recognized as soon as it is chosen**, with no Recognize button, and a click on it shows it as big as the window allows (a `<dialog>`). Adding a person keeps its Upload button, because the name has to be typed first.
- **A refusal stays in the camera's message line for 8 seconds.** The status, asked for every second, carries the camera's own message, so before this a refusal like "No one is registered yet" was replaced within a second.
- **A plain look.** The first one-window page had an indigo-to-violet gradient across the top, an icon in every button, coloured circles with initials, pill-shaped badges and soft shadows: the look of many generated templates, which the user called "vibe coded". It was replaced by a flat one. Thin lines divide the window into panes (no cards, shadows or gradients), buttons are text only, the font is Windows' own (Segoe UI) at 14 pixels, and one neutral set of colours does almost everything, with near-black main buttons in light mode and light ones in dark mode. Colour is kept for what it means: green for a known face or someone present, red for an unknown face, Live and Stop, and blue for links. The camera's area is a quiet grey while the camera is off, and black behind a picture.

**Lessons:**

- **Tested with the real models and a pretend camera.** A demo site, with its data in a temporary folder, registered Colin Powell and Tony Blair from 5 LFW photos each. Its pretend camera showed 5 other LFW photos of Colin Powell, 1.5 seconds each. Live recognition named him (0.76), marked him present, and showed two people in the background as Unknown. A registration through the page took its 5 photos, with the countdown and the pose over the picture.
- **Every form was also tried in a real browser** (the app's browser pane), with a plain picture made by the page itself: recognizing a photo, registering from uploaded photos, and a refused name.
- **A test that waited for the camera to turn off was flaky.** After the fifth registration photo the camera turns off first, and everyone is registered again just after. The test looked at the list of people in between and sometimes didn't find the new person yet. It now waits for the person to appear.
- **`None in photos` fails when `photos` holds numpy pictures**, because `in` compares with `==`, which numpy answers pixel by pixel. The upload route counts the unreadable files instead.
- **Edge in headless mode can't screenshot a page that keeps fetching pictures with `--virtual-time-budget`:** its clock only runs while no request is waiting, so it waited until the camera stopped. The screenshot of a registration in progress was taken in the app's own browser pane instead. Edge also saves its screenshot a few seconds after the command returns. Its `--timeout` waits in real time, but cancels the picture that is loading at that moment, so the camera area came out black. Screenshots of the live camera were later taken through Edge's DevTools protocol instead: a script tells Edge to open the page, waits, and asks for a screenshot. Edge started that way keeps running after its launcher is closed, so the script closes its processes by their profile folder.
- **A small photo stretched to the card's width looked blurry** (a 250-pixel LFW photo), so the CSS only shrinks big photos (`max-width: 100%`) and leaves small ones at their own size.
- **Two copies of the site ran on the same port, and the browser showed the old one.** The step-1 copy, started before the files were rewritten, was never stopped. The complete version started next to it without an error: Flask's server asks to share its port (`SO_REUSEADDR`), and on Windows that lets a second program listen on a port that is already taken. Which of the two gets the browser's requests isn't defined; here it was the old one, so the page still showed step 1. `netstat -ano` showed two programs listening on 127.0.0.1:5000. Stopping the old copy fixed it, and now a second copy refuses to start.
- **What needs a restart after changing the code.** Python code and the HTML template are read once, when the server starts (Flask only re-reads templates in debug mode). The CSS and JavaScript are read from disk at every request and sent with `Cache-Control: no-cache`, which makes the browser check with the server every time whether its copy is still current, so reloading the page is enough.
- **On this computer, the page gets only 1536×696 pixels.** The screen is 1920×1080, but Windows scales everything to 125% (1920 / 1.25 = 1536), and Chrome's tabs, address bar and bookmarks bar, and the taskbar, take the rest of the height. The first layout sized the camera picture by the page's width (4:3 of about 1,000 pixels is 750 pixels tall), so the picture alone was taller than the window, and every feature needed scrolling. The new layout was checked at 1536×696, 1366×625 (a 1366×768 laptop), 1280×562 (a 1080p laptop at 150%), 1920×950, 2560×1300 and 1024×560: none of them scrolls. With the plain look, the picture is 728×546 on this computer and 549×412 at 1280×562, and even at 1024×560 the camera's buttons stay on one row and no name is cut off.
- **A running server keeps the old HTML.** Flask read `index.html` when the page was first asked for and kept it, while the CSS and JavaScript are read at every request. A reload before restarting the server would have mixed the old HTML with the new JavaScript, which looks for parts the old page doesn't have. So after changing the HTML, restart the server before reloading the page.

## Our own SFace

**Goal:** learn how a face recognizer like SFace is made by training one ourselves, and measure how close it gets to OpenCV's. The detector (YuNet) stays as it is. All the code is in `training/`.

**Status:**

- **First model: done, replaced by the second.** Trained on Kaggle (Version 6 of the notebook `our_sface_kaggle`, 13 November 2026: 30 epochs of 5.8 minutes, about 1,400 photos per second on one T4, 3.2 hours of training; files in `training/runs/mobilefacenet_sface/`). The app used it until the second model was ready; `python -m training.export_onnx --run mobilefacenet_sface` brings it back.
- **Second model: done, in use.** Version 7 of the notebook ("v2: 90 epochs + webcam-style photo changes"), 14 November 2026: 90 epochs of 6.8 minutes, about 1,190 photos per second, 10.7 hours in all (38,408 seconds); files in `training/runs/mobilefacenet_sface_v2/`. It trained longer, with the official learning-rate drops at 50, 70 and 80, and with webcam-style changes to the training photos. Exported to `models/our_sface.onnx` (4 MB); it is the app's only recognizer. It beats the first model on the standard LFW test, not in the college test, and it is a little less accurate than OpenCV's pretrained SFace (see "Our models against the target").

### How it works

- **The network: MobileFaceNet** (`training/mobilefacenet.py`), the same small network (1,003,264 weights) as in the official SFace code. It takes an aligned 112×112 face and gives 128 numbers, the same input and output as OpenCV's SFace.
- **Training is "which of the 10,572 people is this?"** Every training person gets a learnable *class centre* (128 numbers). Embeddings and centres are scaled to length 1, so the angle between them says how alike they are.
- **The SFace loss** (`training/sface_loss.py`) pulls each face's embedding towards its own person's centre and pushes it away from everyone else's. Its idea is to weight the pull and push with sigmoid curves of the angle, so they fade out once a face is close enough to its own centre (under a = 0.87 radians, about 50°) and far enough from the others (over b = 1.2 radians, about 69°). Older losses keep pulling and pushing forever, which makes the network memorize noisy, mislabelled web photos. The weights only set how hard to pull or push; no learning flows through them.
- **After training, the class centres are thrown away.** Only the network is kept: for new people it never saw, it still puts photos of the same person close together. That is why it can recognize college students.
- **Training data: CASIA-WebFace**, 490,623 web photos of 10,572 celebrities, already aligned to 112×112 (Kaggle dataset `yakhyokhuja/webface-112x112`, 2.72 GB). It is the dataset of the official SFace code's CASIA-WebFace experiments. Kaggle lists its licence as "Unknown", and the original dataset is for research use only.
- **Test data: LFW** (the figshare copy scikit-learn uses, 181 MB, checked against scikit-learn's published checksums), with every photo aligned by the app's own YuNet and alignment. The stored crops (`datasets/lfw_aligned.npz`) were made with OpenCV's alignCrop, before our own `align_face` replaced it; the two agree to within 1 brightness step (see "Technical decisions" above).

### Technical decisions

- **Settings from the official SFace PyTorch code** (github.com/zhongyy/SFace, `SFace_torch`): s = 64, k = 80, a = 0.87, b = 1.2 (from its CASIA-WebFace command), SGD with momentum 0.9, learning rate 0.1 divided by 10 three times, weight decay 5e-4, batch 256, and photos flipped left-right half the time.
- **How long to train.** The official CASIA-WebFace command trains 125 epochs and drops the learning rate at epochs 50, 70 and 80. The first model trained only 30 epochs, with drops at 16, 24 and 28, to fit a small GPU budget. The second trains 90 epochs with the official drops at 50, 70 and 80 (the `train.py` defaults now). It stops at 90 because after the last drop the learning rate is too small to change much: the first model's last drop gained nothing, and neither did the second's. 90 epochs took 10.7 hours on Kaggle, inside the 12-hour limit of one session.
- **Webcam-style photo changes** (`webcam_style` in `training/datasets.py`, second model only). The training photos are mostly sharp, well-lit celebrity photos, while webcams give small, blurry, dark or washed-out faces. So each training photo is randomly changed: 20% are shrunk to 24–80 pixels and enlarged back (a face far from the camera), 20% are blurred, and 30% get different brightness and contrast. About 45% of photos stay unchanged. `--no-webcam-changes` turns it off.
- **Other changes, for a small GPU budget:** a 1,000-step warm-up from almost 0 up to 0.1; no weight decay on batch-norm values and PReLU slopes (the official code only leaves out batch-norm values); and 16-bit arithmetic. That is bfloat16 on GPUs that have it (RTX 30xx and newer), or float16 with a gradient scaler on Kaggle's T4. Training uses one GPU only, even on Kaggle's "T4 x2" (see lessons).
- **Embeddings of 128 numbers.** The official SFace code uses 512 by default, but OpenCV's SFace gives 128, so ours fits the app and its database unchanged.
- **The network takes raw pixels (0 to 255, RGB order) and scales them itself** (`(pixel - 127.5) / 128`), because that is exactly what OpenCV's `FaceRecognizerSF` passes to a model (see lessons): our model takes the same input as the original SFace. The app runs it with `cv2.dnn` and prepares that same input itself (`run_network`).
- **The training photos are packed into one 2.7 GB file** (`datasets/webface_images.bin`, with `webface_index.npz` saying where each photo starts). Each training process reads it through a memory map, which is faster than opening 490,623 small files.
- **LFW accuracy is measured after every epoch**, so you can watch the network improve. Progress is saved after every epoch (`training/runs/<run>/last.pt`), and `python -m training.train` carries on from there after a stop.
- **Exported with PyTorch's older ONNX exporter** (`dynamo=False`): one 4 MB file, and OpenCV's embeddings equal PyTorch's (similarity 1.000000).
- **Saved networks are loaded with `weights_only=True`**, so loading a file downloaded from Kaggle can only read numbers, never run code hidden in it.
- **Two LFW tests** (`training/evaluate.py`): the standard 6,000-pair test that research papers report, and the college test from "Accuracy measured on the LFW dataset" (100 registered people, 1,000 strangers, averaged over 3 random draws). Each model's threshold is chosen with the same rule that gave SFace its 0.46: the lowest threshold at which at most 0.4% of strangers get a name.
- **Trained on Kaggle**, because this laptop's GPU overheats (see lessons). `training/our_sface_kaggle.ipynb` is built from the project's own files by `training/make_kaggle_notebook.py`, so Kaggle runs exactly the code in this folder.

### Lessons

- **The training photos are aligned the same way the app aligns faces.** OpenCV's alignCrop uses the standard 5-point "ArcFace" template: a 112×112 test image whose landmarks already sat on that template came back unchanged. On 1,000 training photos, YuNet found the eyes and nose within 1–2 pixels of the template, but the mouth corners about 4–5 pixels higher. LFW faces aligned by the app itself showed the same mouth difference, so it is how YuNet places mouth corners, not a different alignment. Training photos and the app's faces match within about 1 pixel. The app's own `align_face`, which replaced alignCrop later, uses the same template.
- **OpenCV gives a recognition model raw RGB pixels.** An untrained network exported to ONNX gave exactly the same embeddings in OpenCV (fed BGR faces, as the app does) as in PyTorch (fed RGB pixels 0–255). So `FaceRecognizerSF` swaps blue and red and doesn't scale the pixels.
- **This laptop's GPU overheats.** The RTX 4050 was already at 81°C before training. During a speed test it reached 89°C and throttled itself to 210 MHz, a small fraction of its normal speed. In a short test it trained about 90 photos per second, so one pass over the training photos would take about 1.5 hours and 30 epochs about two days. Better cooling (a hard surface, a cooling pad, the laptop's Performance fan mode) might make local training possible.
- **PyTorch's newer ONNX exporter crashed on the Windows console.** It prints a symbol (✅) the console can't show; the older exporter works.
- **The whole notebook was rehearsed here** with the 12 people who have the most photos (7,866 photos), laid out the way Kaggle provides them, using float16 like a T4. Two short epochs already lifted LFW accuracy from guessing to 67% and then 71%.
- **The rehearsal was not Jupyter, and missed one Jupyter rule.** On Kaggle, the first run failed after 29 seconds: `%%writefile` refuses a cell with nothing in it, and `app/__init__.py` is an empty file. The notebook now creates empty files with `open(name, "w").close()` instead.
- **Kaggle's dataset folders are slow to read one file at a time, but fine in parallel.** Added with "Add Input", the dataset appears as one folder per person (`/kaggle/input/datasets/yakhyokhuja/webface-112x112/webface_112x112/id_0/0_0.jpg`). Packing photo by photo didn't reach 50,000 photos in 8 minutes: each file takes a while to arrive over the network. Downloading the zip from inside Kaggle instead failed (the public download link returns 404 there). Reading with 64 threads at once fixed it: listing the 10,572 folders took 8 seconds, and packing ran at about 100,000 photos every 4 minutes.
- **Two GPUs were 5.6 times slower than one.** The first full run used both T4s (`torch.nn.DataParallel` splits every batch in half) and trained only 266 photos per second, so 30 epochs would have needed about 15 hours. A separate speed-test notebook on Kaggle (`sface_speed_test`, with made-up images) measured, in photos per second: one T4 with channels_last 1,604; one T4 without it 1,121; both T4s with DataParallel 286. For a network this small, copying it to the second GPU and gathering the results every step costs far more than the second GPU adds. Training now uses one T4. Loading the photos was not the problem: one CPU core of this laptop reads and decodes about 1,500 photos per second.
- **A `!command` that fails doesn't stop a notebook.** Jupyter carries on to the next cell, so a failed step could have ended as a "successful" run with no model. The notebook runs its steps through a small `run()` function that stops it when a step fails.
- **The learning-rate drops did most of the late improvement.** At learning rate 0.1, LFW accuracy climbed fast at first (86.5% after epoch 1, 94.4% after epoch 4), then stalled around 96–96.6% from epoch 9 to 16. The first drop, to 0.01, lifted it to 98.25% in a single epoch, and it reached about 98.7% by epoch 21. The second drop (epoch 25) added a little (98.98%); the third (epoch 29) changed nothing. The whole log is in `training/runs/mobilefacenet_sface/log.csv`.
- **The second model followed the same pattern, stretched over 90 epochs.** At learning rate 0.1 it reached 96% by epoch 9, then crept up to about 97% by epoch 50, a little above the first model's 96–96.6% after its 16 epochs at 0.1, even though its training photos were harder. The first drop (epoch 51) lifted it from 97.25% to 98.67% in one epoch, and the second (epoch 71) from 98.78% to 99.12%. The third (epoch 81) changed nothing, just as in the first model. It finished at 99.13% (on Kaggle's crops). Its accuracy on its own training photos jumped at each drop too: about 83%, then 93%, then 96%. The whole log is in `training/runs/mobilefacenet_sface_v2/log.csv`.
- **Three random draws are too few to compare two close models.** With the 3 draws of `training/evaluate.py`, the second model named 96.3% of registered people against the first model's 96.7%, which looked like a step back. Repeated with 20 draws and thresholds in steps of 0.002 (a one-off script), the two were the same: 96.9% and 97.1% with the mirror trick, at 0.4% of strangers named. Each draw has only about 350 test photos, so 0.4 points is one or two photos per draw. In the college test, differences under about half a point are luck.
- **The same network scored 98.95% on Kaggle and 98.82% here.** PyTorch on this computer also gives 98.82%, so the ONNX export is exact. The difference is the LFW crops: Kaggle aligned LFW with OpenCV 4.14, this computer with OpenCV 5.0, and the two versions place the face slightly differently. Numbers are only comparable when they use the same crops, so every comparison below uses this computer's.
- **The webcam-style photo changes slowed training on Kaggle, though not on this laptop.** Here, one CPU core decoded about 6,000 training photos per second, and 4,000 with the changes: about 0.1 ms more per photo, which looked negligible next to the GPU. On Kaggle, training dropped from about 1,400 to 1,190 photos per second (6.8 minutes per epoch instead of 5.8), with nothing else changed. So Kaggle's CPUs prepare photos more slowly than this laptop's, and with the extra work, preparing photos probably became the bottleneck. 90 epochs still fit in one session (about 10.7 hours), but a speed test on Kaggle itself would have caught this before the run.
- **Chrome saved the downloaded `model.pt` and `last.pt` as `model.zip` and `last.zip`.** A PyTorch `.pt` file is a zip archive inside, and Kaggle's download says so. Renaming them back is all that is needed.

### The target: OpenCV's SFace, measured with `python -m training.evaluate --compare-opencv`

- **Standard LFW test: 99.35%** of the 6,000 pairs right (spread over the 10 folds: 0.37%). The best one-to-one threshold was about 0.31, close to the 0.363 of OpenCV's example.
- **College test** (100 registered people, 3 random draws). It agrees with the earlier measurement in "Accuracy measured on the LFW dataset", which used other random draws, so both tests are measuring the same thing:

| Photos per person | Threshold | Registered people named correctly | Strangers given a name |
| --- | --- | --- | --- |
| 5 | 0.363 | 100% | 13.7% |
| 5 | 0.40 | 99.8% | 3.5% |
| 5 | 0.45 | 99.6% | 0.4% |
| 5 | **0.46** | 99.4% | 0.2% |
| 5 | 0.50 | 99.0% | 0.1% |
| 3 | 0.45 | 99.3% | 0.4% |
| 1 | 0.45 | 95.2% | 0.2% |

- The threshold rule (at most 0.4% of strangers named) gives 0.45 with these draws, next to the 0.46 the app uses, so 0.46 stays.

### Our models against the target

Measured with `training/evaluate.py` on the same LFW crops and the same random draws as OpenCV's SFace above, except the summary table, which uses more draws.

**Summary.** Each model at its own threshold, chosen by the rule (at most 0.4% of strangers named). The college-test columns use 20 random draws and thresholds in steps of 0.002, from a one-off script, because the 3 draws of `training/evaluate.py` are too few to separate our two models (see lessons):

| Model | Standard LFW test | Threshold | Named, 5 photos | Named, 3 photos | Named, 1 photo | Named at 1% of strangers (5 photos) |
| --- | --- | --- | --- | --- | --- | --- |
| OpenCV's SFace | 99.35% | 0.454 | 99.5% | 99.3% | 95.6% | 99.6% |
| First model | 98.82% | 0.478 | 96.9% | 95.0% | 75.3% | 97.7% |
| First model, mirror trick | 98.95% | 0.480 | 97.1% | 95.4% | 76.7% | 97.8% |
| Second model | 99.13% | 0.478 | 97.0% | 95.3% | 76.8% | 98.0% |
| **Second model, mirror trick** (the app) | 99.13% | 0.482 | 96.9% | 95.4% | 77.1% | 98.1% |

**Second model, college test** with the mirror trick, as the app uses it (`python -m training.evaluate`, 3 random draws):

| Photos per person | Threshold | Registered people named correctly | Strangers given a name |
| --- | --- | --- | --- |
| 5 | 0.363 | 99.5% | 11.6% |
| 5 | 0.40 | 99.3% | 4.0% |
| 5 | 0.46 | 97.3% | 0.7% |
| 5 | **0.48** | 96.3% | 0.3% |
| 5 | 0.50 | 95.3% | 0.1% |
| 3 | 0.48 | 95.2% | 0.3% |
| 1 | 0.48 | 74.9% | 0.1% |

**First model, college test** without the mirror trick (3 random draws):

| Photos per person | Threshold | Registered people named correctly | Strangers given a name |
| --- | --- | --- | --- |
| 5 | 0.363 | 99.4% | 11.5% |
| 5 | 0.40 | 98.7% | 4.1% |
| 5 | 0.46 | 97.0% | 0.8% |
| 5 | **0.49** | 95.9% | 0.2% |
| 5 | 0.50 | 95.1% | 0.2% |
| 3 | 0.49 | 94.0% | 0.3% |
| 1 | 0.49 | 68.1% | 0.0% |

- **Standard LFW test:** the second model gets 99.13% of the 6,000 pairs right, with or without the mirror trick (spread over the 10 folds: 0.33% without it, 0.42% with it). The first model got 98.82% (0.31%), or 98.95% with the mirror trick. The best one-to-one threshold was about 0.25 for both.
- **The threshold rule gives 0.48 for the second model,** with or without the mirror trick, so the app keeps 0.48. For the first model it gave 0.49 without the mirror trick and 0.48 with it. SFace's 0.46 would give 0.7–0.8% of strangers a name with either of ours.
- **Three times the training improved the standard test, not the college test.** On the standard test the second model makes about 52 mistakes in 6,000 pairs where the first made about 71, a quarter fewer. In the college test, at 0.4% of strangers named, both name about 97% of registered people. The second model is only a little better with 1 photo per person and at a looser 1% of strangers. So the extra training fixed some of the moderately hard pairs, the ones that matter at the standard test's low threshold (about 0.25). The strict threshold of the college test is decided by the few hardest cases: strangers who look like someone in the class, and photos of a registered person that look unlike their others. The extra training didn't fix those.
- **Side by side with OpenCV's SFace, at each model's own threshold:** both give about 0.4% of strangers a name, but ours misses more registered people. With 5 photos per person ours names about 97% (OpenCV's: 99.5%), with 3 photos 95.4% (99.3%), and with 1 photo only about 77% (95.6%). In a class registered with 5 photos each, about 3 in 100 students would show up as "Unknown" with our model, against about 1 in 200 with OpenCV's SFace. Registering 5 photos matters even more with our model.
- **Our models' strangers spread higher.** One-to-one, ours need a lower threshold than OpenCV's (0.25 against 0.31), but one-to-many they need a higher one (0.48 against 0.45–0.46): a few strangers get unusually high similarities to someone in the class, so the threshold has to sit higher, where it also cuts off more of the registered people.
- **On the 5 webcam photos in `data/people/aadi2`,** each photo was compared with the average of the other four. OpenCV's SFace: 0.660, 0.737, 0.630, 0.539, 0.287. First model: 0.519, 0.649, 0.534, 0.293, 0.211. Second model: 0.407, 0.639, 0.535, 0.378, 0.234. Both of ours are lower than OpenCV's on every photo, and the gap is biggest on photo 4 (by their order, the "tilt up" pose). The second model does better than the first on the two hardest photos (4 and 5) and worse on photo 1, and the same on average (0.439 against 0.441). OpenCV's reaches its threshold on 4 of the 5 photos, the first model on 3, the second on 2. So the webcam-style photo changes did not visibly help on these photos. One person and 5 photos are only a hint, not a measurement.
- **The mirror trick** (`python -m training.evaluate --mirror on`). For the first model it helped a little: the standard test went from 98.82% to 98.95%, and in the 3-draw college test, at its rule thresholds (0.49 without, 0.48 with), 96.7% were named instead of 95.9% with 5 photos, 95.2% instead of 94.0% with 3, and 73.0% instead of 68.1% with 1. For the second model it makes no measurable difference with 3–5 photos (99.13% on the standard test either way) and helps a little with 1 photo (77.1% against 76.8% over 20 draws, 74.9% against 74.1% over 3). On the webcam photos it raised each of the second model's 5 similarities a little (to 0.452, 0.641, 0.548, 0.395, 0.238); whether strangers rise as much there can't be told from one person. For OpenCV's SFace it made no difference (99.43% on the standard test, and 99.6% named at 0.4% strangers, as without it). Our model is small, so the extra cost is only about 3 ms per face (8 ms with OpenCV's). So the app uses it, at threshold 0.48.
- **Why ours is behind, as far as we can tell.** OpenCV's model card says its SFace models are "MobileFaceNet instances trained on the SFace loss function", the same network and loss as ours. It doesn't say what data they were trained on. Training length was the one known difference: the official CASIA-WebFace command trains for 125 epochs, with the learning rate dropping at epochs 50, 70 and 80. The second model trained 90 epochs with those same drops, and stopped improving after the last one, as the first model had; the official code's remaining 35 epochs run at that same smallest learning rate. Its college-test gap to OpenCV's didn't shrink, so training length is not what holds ours back there. The likeliest remaining difference is the training photos. The official SFace code also supports VGGFace2 and MS1MV2, which are much bigger than CASIA-WebFace (its training script is set to MS1M by default), but which data OpenCV's model used is not known.

### Training on Kaggle

The notebook is on Kaggle as the private notebook `our_sface_kaggle` in your account. To run it again, or to set it up in another account:

1. Sign in at kaggle.com and verify your phone number in your account settings. Kaggle only gives GPUs and internet access to verified accounts.
2. Create a new notebook, then use **File → Import Notebook** to upload `training/our_sface_kaggle.ipynb`. In the import dialog's **Advanced Settings**, choose **Quick Save**, so it doesn't run straight away without a GPU. In the editor, **File → Import Notebook** replaces the notebook with a newer copy of the file.
3. In the right-hand panel, **Add Input**: paste `https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112` into the search box and add "WebFace 112x112". (Searching for the name alone mostly found other people's notebooks.)
4. In the notebook's settings, set **Accelerator** to **GPU T4 x2** and turn **Internet** on.
5. **Save Version → Save & Run All (Commit)**. It runs in the background, so you can close the browser. Kaggle gives 30 free GPU hours a week and stops a single run after 12 hours. Training stops itself after the epoch that would pass 11.5 hours from the start of the notebook, and still saves the model.
6. When it has finished, open that version's **Output**, go to `training/runs/mobilefacenet_sface_v2/` (`mobilefacenet_sface/` for the first model), select each file and click its download icon: `model.pt` (4.2 MB), `last.pt` (19 MB, only needed to carry on training) and `log.csv`. Chrome saves the `.pt` files as `.zip`; rename them back. Clicking `log.csv` opens it as text in a new tab instead; save that page. Put the three files in the folder of the same name in this project's `training/runs/`.
7. Run `python -m training.export_onnx`, then `python -m training.evaluate`. Set `MATCH_THRESHOLD` in `app/config.py` to the threshold the evaluation chose. If `data/face_database_ours.json` exists, delete it: it holds the old model's numbers, and the app only registers everyone again when that file is missing.

## Known limitations

- **No liveness check.** A printed photo, or a phone showing a registered person's face, is recognized as that person, so attendance can be faked. Real systems add anti-spoofing for this.
- Works best with a face looking at the camera in good light. Side views, masks and dark rooms give lower similarity, which means more "Unknown". The strict threshold (0.48) makes this a little more frequent, on purpose: "Unknown" is better than a wrong name.
- The accuracy numbers come from news photos (LFW), not webcam pictures. Accuracy on a real webcam hasn't been measured.
- Our own SFace misses more registered people than OpenCV's pretrained SFace would: about 3 in 100 with 5 photos each (OpenCV's: about 1 in 200), and about 1 in 4 with a single photo (see "Our models against the target"). So register 5 photos per person.
- Only the recognizer is our own. The face detector, YuNet, is OpenCV's pretrained model.
- Names are drawn with OpenCV's built-in font, which only has English letters, digits and basic symbols.
- A camera that only Media Foundation can open, not DirectShow (some newer built-in laptop cameras), isn't in the Camera list. When the list is empty, it offers the plain numbers Camera 0, 1 and 2 instead, so such a camera can still be tried.
- Close `attendance.csv` in Excel before starting live recognition. Excel locks the file on Windows, and then the program can't write to it.
- The attendance date is set when live recognition starts, so leaving it running past midnight keeps the old date. In the window it is set when the window opens.
- Only the webcam has a thread of its own. Everything else in the window runs in one thread, so it stops responding for a moment while the database is rebuilt (longer with many photos).
- The web version uses Flask's built-in development server. It is made for one computer, like here, not for putting a site on the internet.
- The web version and the window don't see each other's changes while both are running: someone registered in the window only shows up in the web version after "Register everyone from the photo folders" or a restart, and the other way round. Only one of them can use a camera at a time.
- Every browser tab of the web version shows the same camera: the camera, a registration and the attendance belong to the program, not to a tab.
- The web version was tested with the real models and a pretend camera, and in a real browser, but not yet with a real camera.
- If a webcam never finishes opening (a stuck camera driver), Python can't stop its thread. Stop camera brings the window back, but the webcam can only be used again after restarting the program; the message says so after 20 seconds.

## Privacy

Face photos and embeddings are personal (biometric) data. Get each person's consent before registering them, and keep `data/` on this computer; `.gitignore` already keeps it out of git.

The training and test sets in `datasets/` (CASIA-WebFace and LFW) are photos of real people collected from the web, shared for research. Use them only for learning and research, don't share them further, and delete `datasets/` when they are no longer needed. `.gitignore` keeps them and `training/runs/` out of git.
