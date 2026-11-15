# How to run

The short version. Every detail is in [README.md](README.md).

## 1. Install (once)

1. Install **Python 3.10 or newer** from [python.org](https://www.python.org/downloads/). In the installer, tick **Add python.exe to PATH**.
2. Download this project: on GitHub, click **Code → Download ZIP** and unzip it.
3. Open a terminal in the project folder (in File Explorer, right-click inside the folder → **Open in Terminal**) and run:

   ```bash
   pip install -r requirements.txt
   ```

## 2. Start it

```bash
python web_app.py
```

Your browser opens `http://127.0.0.1:5000`. The first start downloads a small model that finds faces (0.2 MB), so it needs the internet once. Keep the terminal open while you use it, and press **Ctrl+C** in it to stop.

## 3. Use it

1. **Choose the camera** in the list above the picture.
2. **Register each person.** Under **Add person**, type the name and click **Take 5 photos with the camera**, then follow the pose and the countdown on the picture. Ask people before registering them: face photos are personal data.
3. **Take attendance.** Click **Start live attendance**. Everyone recognized appears under **Present today**. **Stop** turns the camera off.
4. **Get the register.** **Download CSV** gives today's attendance; **History** shows any day.

## If something goes wrong

- **"The web version is already running"**: it is still open in another terminal. Use that one, or press Ctrl+C there first.
- **The camera doesn't open**: close other apps using it (Camera, Zoom, Teams, a browser tab), turn on "Let desktop apps access your camera" in Windows Settings → Privacy & security → Camera, or choose another camera and click **Refresh**.
- **`python` or `pip` is not recognized**: install Python again with **Add python.exe to PATH** ticked, then open a new terminal.

Prefer a desktop window to the browser? Run `python ui.py` instead: it does the same things.
