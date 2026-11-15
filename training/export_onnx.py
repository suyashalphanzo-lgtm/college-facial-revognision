"""Saving our trained network as a file the app can load.  Run:
    python -m training.export_onnx

ONNX is a common file format for neural networks. OpenCV can run ONNX files but not PyTorch ones,
so this is how our model gets into the app. It is saved as models/our_sface.onnx and then checked:
OpenCV, run the way the app runs it, must give the same embeddings as PyTorch does.
"""

import argparse
import warnings

import numpy as np
import torch

from app import config
from training.evaluate import embed_with_opencv, embed_with_torch, load_lfw
from training.mobilefacenet import MobileFaceNet
from training.train import DEFAULT_RUN, RUNS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN, help="folder name in training/runs/ (the first model: mobilefacenet_sface)")
    parser.add_argument("--untrained", action="store_true", help="export a fresh network (to test the export)")
    args = parser.parse_args()

    model = MobileFaceNet()
    if not args.untrained:
        run_dir = RUNS_DIR / args.run
        finished = run_dir / "model.pt"
        # weights_only: load numbers only, never code (safe for files downloaded from Kaggle).
        if finished.exists():
            state = torch.load(finished, weights_only=True)
        else:
            state = torch.load(run_dir / "last.pt", weights_only=True)["model"]
        model.load_state_dict(state)
        print(f"Exporting {'the finished network' if finished.exists() else 'the last saved epoch'} of {args.run}")
    model.eval()  # batch normalization uses its learned averages, not the current batch

    config.MODELS_DIR.mkdir(exist_ok=True)
    # PyTorch's older exporter (dynamo=False) writes a single file that OpenCV reads exactly right.
    # It is marked as deprecated, so its warning is hidden. The newer exporter crashed on this
    # Windows console, and by default it puts the weights in a second file.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model,
            (torch.zeros(1, 3, 112, 112),),
            str(config.OUR_MODEL_FILE),
            dynamo=False,
            input_names=["face"],
            output_names=["embedding"],
            opset_version=17,
        )
    print(f"Saved {config.OUR_MODEL_FILE} ({config.OUR_MODEL_FILE.stat().st_size / 1e6:.1f} MB)")

    crops = load_lfw()[0][:300]
    similarity = np.sum(embed_with_opencv(config.OUR_MODEL_FILE, crops) * embed_with_torch(model, crops), axis=1)
    print(f"OpenCV vs PyTorch on 300 LFW faces: lowest similarity {similarity.min():.6f} (should be 1.000000)")
    if similarity.min() < 0.9999:
        raise SystemExit("OpenCV's embeddings differ from PyTorch's: the exported file is not usable.")


if __name__ == "__main__":
    main()
