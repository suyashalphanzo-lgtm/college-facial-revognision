"""Builds training/our_sface_kaggle.ipynb, a notebook that trains our SFace on Kaggle's free GPUs.  Run:
    python -m training.make_kaggle_notebook

The notebook carries the project's own files (copied in below with %%writefile), so Kaggle runs exactly
the code in this folder. Run this script again after changing any of those files.
How to use the notebook: see "Training on Kaggle" in CLAUDE.md.
"""

import json

from app.config import PROJECT_DIR
from training.train import DEFAULT_RUN

NOTEBOOK_FILE = PROJECT_DIR / "training" / "our_sface_kaggle.ipynb"
PROJECT_FILES = [
    "app/__init__.py",
    "app/config.py",
    "app/face_engine.py",
    "training/__init__.py",
    "training/mobilefacenet.py",
    "training/sface_loss.py",
    "training/datasets.py",
    "training/evaluate.py",
    "training/train.py",
]


def cell(kind: str, text: str) -> dict:
    if not text.startswith("%%writefile"):  # the project's own files are copied unchanged
        text = text.replace("RUN_NAME", DEFAULT_RUN)  # the run folder that training/train.py uses
    lines = text.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def main() -> None:
    cells = [
        cell("markdown", """# Training our own SFace

This notebook trains the face recognizer of the college face recognition project on Kaggle's free GPU.
Before running it:

1. **Add Input** (right-hand panel): add the dataset `https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112`.
2. **Settings → Accelerator**: `GPU T4 x2`. **Settings → Internet**: on (to download LFW and the YuNet model).
3. **Save Version → Save & Run All (Commit)**. It runs in the background, so the browser can be closed.

When it has finished, download `training/runs/RUN_NAME/` from the version's **Output** tab
and put it in the project's `training/runs/` folder.
"""),
        cell("code", """!nvidia-smi
import os, shutil, subprocess, time
from pathlib import Path
import cv2, torch

NOTEBOOK_STARTED = time.time()
print("PyTorch", torch.__version__, "| OpenCV", cv2.__version__, "|", torch.cuda.device_count(), "GPU(s)")


def run(command):
    \"\"\"Run a command, show its output, and stop the notebook if it fails (a plain !command would carry on).\"\"\"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"Failed: {command}")
"""),
        cell("markdown", "The project's own files, exactly as they are in the project folder:"),
        cell("code", "import os\nos.makedirs('app', exist_ok=True)\nos.makedirs('training', exist_ok=True)\n"),
    ]
    for name in PROJECT_FILES:
        text = (PROJECT_DIR / name).read_text(encoding="utf-8")
        if text:
            cells.append(cell("code", f"%%writefile {name}\n" + text))
        else:  # %%writefile refuses an empty cell, so empty files are made with open() instead
            cells.append(cell("code", f'open("{name}", "w").close()  # an empty file\n'))

    cells += [
        cell("markdown", "Find the training photos (the dataset added with **Add Input**):"),
        cell("code", """# The packed photos (2.7 GB) go to /tmp, which is not saved with the notebook's output.
os.environ["FACE_DATASETS_DIR"] = "/tmp/datasets"
found = None
for depth in range(1, 6):  # the folder of person folders (id_0, id_1, ...) can sit a few levels down
    found = next((path for path in Path("/kaggle/input").glob("*/" * depth + "id_*") if path.is_dir()), None)
    if found:
        break
assert found, "Add the dataset 'webface-112x112' by yakhyokhuja with 'Add Input' first."
WEBFACE = found.parent
print(WEBFACE)
"""),
        cell("markdown", "Pack the training photos into one file, and download and align LFW for the test after each epoch:"),
        cell("code", 'run(f\'python -m training.datasets --webface "{WEBFACE}"\')\n'),
        cell("markdown", """Train. Every epoch prints LFW accuracy and saves `training/runs/RUN_NAME/last.pt`.
Kaggle stops a run after 12 hours, so training ends early (and still saves the model) if it would run past 11.5 hours
from the start of this notebook:"""),
        cell("code", """hours_left = 11.5 - (time.time() - NOTEBOOK_STARTED) / 3600
run(f"python -m training.train --max-hours {hours_left:.2f}")
"""),
        cell("markdown", "Results. The downloaded models are deleted so the output only holds what is needed:"),
        cell("code", """shutil.rmtree("models", ignore_errors=True)
run_dir = Path("training/runs/RUN_NAME")
print((run_dir / "log.csv").read_text())
for path in sorted(run_dir.iterdir()):
    print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")
"""),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    NOTEBOOK_FILE.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Saved {NOTEBOOK_FILE}")


if __name__ == "__main__":
    main()
