# College Face Recognition

A face recognition system for college use. It registers people's faces, recognizes them in photos or live on a webcam, and keeps an attendance register. It was built step by step as a learning project. Faces are recognized by our own model: a copy of the SFace face recognizer that we trained from scratch on Kaggle's free GPU. Only the part that finds faces in a picture is OpenCV's pretrained YuNet.

- **How to run it, in short:** [HOW_TO_RUN.md](HOW_TO_RUN.md). Every detail: this file.
- **How the whole system works, every decision and every measurement:** [CLAUDE.md](CLAUDE.md), the project's technical textbook.
- **How we built and trained our own face recognition model, step by step:** [BUILD_LOG.md](BUILD_LOG.md).

## What it does

1. **Detection:** finds every face in a picture (OpenCV's YuNet model).
2. **Embedding:** lines each face up (eyes level, always the same size) and turns it into 128 numbers with our own SFace model. Photos of the same person give similar numbers.
3. **Registration:** remembers each registered person as the average of their photos' numbers.
4. **Matching:** names a face when its numbers are close enough to a registered person's, or shows "Unknown".

On top of this, a person recognized in 10 webcam frames in a row is marked present, once per day.

## Setup

1. Install Python 3.10 or newer from python.org. It was built on Windows 11 with Python 3.13. Everything except the two "Open ..." buttons also works on other systems.
2. Download the project. On GitHub, click **Code → Download ZIP** and unzip it, or run:

   ```bash
   git clone https://github.com/Aaditya13050/college-face-recognition.git
   ```

3. Open a terminal in the project folder and install the libraries (OpenCV, NumPy, Pillow, pytest, Flask):

   ```bash
   pip install -r requirements.txt
   ```

Our own model comes with the project (`models/our_sface.onnx`). The first start downloads OpenCV's face finder, YuNet (0.2 MB), into `models/`. After that, no internet is needed.

## How to run

### The window (recommended)

```bash
python ui.py
```

First choose the camera in the **Camera** list under the picture. It shows every camera Windows knows, by name: for example a laptop's built-in webcam, a phone used as a webcam (like Iriun), or OBS Virtual Camera. Click **Refresh** after plugging a camera in. The app remembers your choice, and you can switch cameras while the camera is running.

1. **Register each person.** Type the name and click **Take 5 photos with the webcam**. The text under the picture says which pose to make (look straight, turn slightly left, turn slightly right, tilt up, tilt down). It counts down 3 seconds before each photo and finishes by itself after the fifth.
   Or click **Open the photos folder**, make a folder named after each person with 3–5 photos of them, and click **Register photos from the folders**.
2. **Live recognition and attendance.** Click **Start live camera + attendance**. Names appear on the picture, and recognized people are added to "Present today" and to `data/attendance.csv`. **Stop camera** turns it off, and **Open the attendance file** opens the register.
3. **Recognize a photo.** Click **Recognize faces in a photo...** and choose a photo. The labelled result is shown, and a full-size copy is saved in `data/results/`.

### The web version

```bash
python web_app.py
```

It opens `http://127.0.0.1:5000` in your browser. It runs only on this computer, and no other device can open it. Keep the terminal open while you use it, and press Ctrl+C there to stop it. It does everything the window does, on one page that fits in the window, so nothing needs scrolling: the camera on the left, the tools in the middle, and today's attendance and the registered people on the right.

- **Camera:** choose the camera above the picture. **Start live attendance** shows the camera with every face boxed and named, and adds people to **Present today** once they have been recognized in 10 pictures in a row. **Stop** turns it off.
- **Add person:** type the name, then either click **Take 5 photos with the camera** (the picture says which pose to make and counts down before each photo), or choose 3–5 photos and click **Upload and register**. Photos copied into `data/people` by hand are registered with **Register everyone from the photo folders**.
- **Recognize:** choose a photo. It comes back at once with every face boxed and named (click it to see it bigger), and a full-size copy is saved in `data/results/`.
- **History:** choose a date, or click one of the recent days, to see who was present. Download that day, or everything, as a CSV file.
- **Registered people:** the × next to a name deletes that person and their photos.

Use either the web version or the window, not both at once: a camera can only be used by one program at a time.

Only one copy of the web version can run at a time. If it says it is already running, open the address in your browser, or stop the other copy first. After changing its Python code or HTML, stop it with Ctrl+C and start it again; after changing its CSS or JavaScript, reloading the page is enough.

### The text menu

```bash
python main.py
```

It uses the camera last chosen in the window, and does the same with typed numbers: 1 registers a person with the webcam (press SPACE for each of the five photos), 2 registers everyone from the photo folders, 3 recognizes a photo (type its path), 4 is live recognition with attendance (press Q to stop), 5 exits.

### The tests

```bash
python -m pytest
```

There are 50 checks for matching, the database, attendance, name checking, the face alignment, the camera list, the webcam thread (with a pretend camera), every part of the web version, and our own network and its loss. They need no camera and no models. Use `python -m pytest`, not plain `pytest`, so Python can find the `app` package.

The 4 checks of our network itself need PyTorch, which only training uses (`pip install -r training/requirements.txt`). Without it, pytest skips that file and reports "46 passed, 1 skipped", which is fine.

## Our own face recognition model

`models/our_sface.onnx` (4 MB) is our own copy of SFace: the MobileFaceNet network trained with the SFace loss on CASIA-WebFace (490,623 photos of 10,572 people), 90 epochs on Kaggle's free GPU. [BUILD_LOG.md](BUILD_LOG.md) tells how it was built, and `training/` holds all the code.

Measured on the LFW face dataset (see [CLAUDE.md](CLAUDE.md)):

| Test | Result |
| --- | --- |
| Standard LFW test (6,000 pairs of photos: same person or not?) | 99.13% correct |
| 100 registered people with 5 photos each: their other photos named correctly | 96.3% |
| 1,000 strangers given a registered person's name | 0.3% |

For comparison, OpenCV's pretrained SFace scores 99.35% and 99.4% in the same tests (`python -m training.evaluate --compare-opencv` measures both). The app doesn't use it.

The match threshold (0.48) and the other settings are in `app/config.py`. The camera is chosen in the window.

## Tips

- Register 5 photos per person, from different angles (the window's "Take 5 photos with the webcam" does this). People registered with only 1 photo are missed about 1 time in 4.
- Works best with a face looking at the camera in good light.
- Close `attendance.csv` in Excel before starting live recognition. Excel locks the file, and then the program can't write to it.
- If the webcam doesn't open (the message under the picture counts the seconds while it tries):
  - Close other apps that use it: Camera, Zoom, Teams, a browser tab.
  - In Windows Settings → Privacy & security → Camera, turn on "Camera access" and "Let desktop apps access your camera".
  - Some laptops have a camera switch, a sliding cover, or a camera key on the keyboard that turns it off.
  - Choose another camera in the Camera list. A virtual camera like OBS shows a black or logo picture when no program is sending it video.
- There is no liveness check: a printed photo of a registered person is recognized as that person.

## Project layout

| Path | What it is |
| --- | --- |
| `ui.py` | The window |
| `web_app.py` | The web version: a small Flask website that runs on this computer |
| `app/camera_session.py` | The web version's camera, registration and live attendance, kept running on the server |
| `web/templates/index.html`, `web/static/style.css`, `web/static/app.js` | The web version's page (HTML), its look (CSS) and its behaviour (JavaScript) |
| `main.py` | The text menu |
| `app/config.py` | Every setting |
| `app/face_engine.py` | Detection, alignment and embeddings (our model runs here) |
| `app/face_database.py` | Registration and the database file |
| `app/recognizer.py` | Matching and drawing names |
| `app/attendance.py` | The attendance rules and the CSV file |
| `app/photo.py`, `app/webcam.py` | Recognizing a photo; the webcam screens |
| `app/cameras.py` | Asks Windows for the cameras' names, for the Camera list |
| `tests/` | The tests |
| `training/` | The code that trained our own SFace (see [BUILD_LOG.md](BUILD_LOG.md)) |
| `training/runs/*/log.csv` | The epoch-by-epoch training logs of our two models |
| `models/our_sface.onnx` | Our trained model (4 MB) |
| `data/` | Created when used: photos of registered people, the databases, attendance, results |

To study the code, follow the "Study order" table in [CLAUDE.md](CLAUDE.md).

## Privacy

Face photos and their numbers are personal (biometric) data. Ask each person before registering them. Nothing in `data/` is uploaded to GitHub: `.gitignore` keeps it on the computer.

## Credits

- The YuNet face detector: the [OpenCV model zoo](https://github.com/opencv/opencv_zoo).
- Training settings for our own SFace: the official SFace code ([zhongyy/SFace](https://github.com/zhongyy/SFace)).
- Our model was trained on CASIA-WebFace and tested on LFW. Both are photo collections shared for research, so our model is for learning and research only. The datasets themselves are not in this repository.
