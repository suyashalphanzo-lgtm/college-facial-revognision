"""Training our own SFace.  Run from the project folder:
    python -m training.train              start, or carry on from the last finished epoch
    python -m training.train --steps 300  a quick speed test; saves nothing

One pass over all 490,623 training photos is an "epoch". After every epoch the model is tested on
the 6,000 LFW pairs and saved to training/runs/<run>/last.pt, so stopping (Ctrl+C, a restart)
loses at most one epoch. When all epochs are done, the finished network is saved as model.pt.

Settings come from the official SFace code (MobileFaceNet, SFace loss, plain SGD with momentum,
learning rate 0.1 divided by 10 at epochs 50, 70 and 80, weight decay 5e-4, batch 256). Changes to
fit a small GPU budget: 90 epochs instead of 125, a short warm-up, and 16-bit arithmetic (see CLAUDE.md).
The first model (run "mobilefacenet_sface") was trained with --epochs 30 --lr-drops 16 24 28 and
without the webcam-style photo changes (training/datasets.py).
The same file runs on this laptop and on Kaggle (training/our_sface_kaggle.ipynb).
"""

import argparse
import csv
import os
import time

import torch

from app.config import PROJECT_DIR
from training.datasets import LFW_ALIGNED, LFW_PAIRS, WebFaceDataset, load_lfw_pairs
from training.evaluate import embed_with_torch, lfw_verification, load_lfw
from training.mobilefacenet import EMBEDDING_SIZE, MobileFaceNet
from training.sface_loss import SFaceLoss

RUNS_DIR = PROJECT_DIR / "training" / "runs"
# The second model: 90 epochs with webcam-style photo changes. The first one is in runs/mobilefacenet_sface.
DEFAULT_RUN = "mobilefacenet_sface_v2"


def learning_rate(args, epoch: int, step: int) -> float:
    """0.1 at the start, divided by 10 at each epoch in args.lr_drops.
    During the first args.warmup_steps steps it climbs from almost 0 to 0.1, so the big early
    updates don't throw the freshly started network off course."""
    rate = args.lr * 0.1 ** sum(epoch >= drop for drop in args.lr_drops)
    if step < args.warmup_steps:
        rate *= (step + 1) / args.warmup_steps
    return rate


def parameter_groups(*modules):
    """Weight decay (gently shrinking weights towards 0, which stops them from memorizing the
    training photos) only for the big weight tables, not for batch-norm values or PReLU slopes."""
    decay, no_decay = [], []
    for module in modules:
        for parameter in module.parameters():
            (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return decay, no_decay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN, help="folder name in training/runs/")
    # The official code drops the learning rate at epochs 50, 70 and 80 of 125. After 80 it barely
    # changes anything (our first run gained nothing from its last drop), so we stop at 90.
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--lr-drops", type=int, nargs="+", default=[50, 70, 80])
    parser.add_argument("--no-webcam-changes", action="store_true", help="train on the clean photos only, like the first run")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count()), help="processes that read photos")
    parser.add_argument("--steps", type=int, default=0, help="stop after this many steps (speed test)")
    parser.add_argument("--precision", choices=["auto", "bfloat16", "float16"], default="auto")
    parser.add_argument("--max-hours", type=float, default=0,
                        help="stop after the epoch that would run past this many hours (Kaggle stops at 12)")
    args = parser.parse_args()
    session_started = time.time()

    torch.manual_seed(1337)
    torch.backends.cudnn.benchmark = True  # lets cuDNN pick the fastest way to run each layer
    device = torch.device("cuda")

    # 16-bit numbers halve the GPU's work. Newer GPUs (RTX 30xx and later, compute capability 8+)
    # have bfloat16, which has the same range as normal 32-bit numbers. Older ones like Kaggle's T4
    # only have float16, whose small range can round tiny gradients to 0, so the loss is first
    # multiplied by a large number and the gradients divided by it again (the GradScaler).
    if args.precision == "auto":
        args.precision = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
    sixteen_bit = torch.bfloat16 if args.precision == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "float16")
    print(f"GPU: {torch.cuda.get_device_name()} (1 of {torch.cuda.device_count()} used), {args.precision}")

    dataset = WebFaceDataset(webcam_changes=not args.no_webcam_changes)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    # Only one GPU is used, even on Kaggle's "T4 x2". Splitting each batch over both T4s with
    # torch.nn.DataParallel was measured at 286 photos/s against 1,604 on a single T4: for a
    # network this small, copying it to the second GPU every step costs more than it saves.
    # channels_last (storing each pixel's channels together) was faster too: 1,604 against 1,121.
    model = MobileFaceNet().to(device).to(memory_format=torch.channels_last)
    loss_function = SFaceLoss(EMBEDDING_SIZE, dataset.num_people).to(device)
    decay, no_decay = parameter_groups(model, loss_function)
    optimizer = torch.optim.SGD(
        [{"params": decay, "weight_decay": 5e-4}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        momentum=0.9,
    )
    print(f"{len(dataset):,} photos of {dataset.num_people:,} people, "
          f"{sum(p.numel() for p in model.parameters()):,} weights in the network")

    run_dir = RUNS_DIR / args.run
    checkpoint_file = run_dir / "last.pt"
    first_epoch, step = 0, 0
    if checkpoint_file.exists() and not args.steps:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        loss_function.load_state_dict(checkpoint["loss_function"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        first_epoch, step = checkpoint["epoch"] + 1, checkpoint["step"]
        print(f"Carrying on from the end of epoch {first_epoch}")

    has_lfw = LFW_ALIGNED.exists() and LFW_PAIRS.exists()
    if has_lfw:
        lfw_crops, _, lfw_stems = load_lfw()
        lfw_pairs = load_lfw_pairs(lfw_stems)
    else:
        print("No aligned LFW found: the LFW test after each epoch is skipped.")

    for epoch in range(first_epoch, args.epochs):
        model.train()
        started = time.time()
        totals = {"loss": 0.0, "accuracy": 0.0, "batches": 0}
        window_started, window_photos = time.time(), 0
        for faces, labels in loader:
            rate = learning_rate(args, epoch, step)
            for group in optimizer.param_groups:
                group["lr"] = rate

            faces = faces.to(device, non_blocking=True).float().contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            # The network runs in 16-bit; the loss itself is computed in full 32-bit precision.
            with torch.autocast("cuda", dtype=sixteen_bit):
                embeddings = model(faces)
            loss, cosines = loss_function(embeddings.float(), labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()  # without float16 the scaler does nothing
            scaler.step(optimizer)
            scaler.update()

            step += 1
            window_photos += len(labels)
            if step % 200 == 0:
                loss_value = loss.item()
                # "Training accuracy": how often the nearest class centre is the right person.
                accuracy = (cosines.argmax(dim=1) == labels).float().mean().item()
                totals["loss"] += loss_value
                totals["accuracy"] += accuracy
                totals["batches"] += 1
                speed = window_photos / (time.time() - window_started)
                print(f"epoch {epoch + 1} step {step} loss {loss_value:.2f} training accuracy {accuracy:.1%} "
                      f"learning rate {rate:.4f} {speed:.0f} photos/s", flush=True)
                window_started, window_photos = time.time(), 0
            if args.steps and step >= args.steps:
                print("Speed test finished.")
                return

        minutes = (time.time() - started) / 60
        lfw_accuracy = float("nan")
        if has_lfw:
            lfw_accuracy, _, _ = lfw_verification(embed_with_torch(model, lfw_crops), *lfw_pairs)
        print(f"===== epoch {epoch + 1} done in {minutes:.1f} min, LFW accuracy {lfw_accuracy:.2%}", flush=True)

        run_dir.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_file.with_suffix(".tmp")
        torch.save(
            {"model": model.state_dict(), "loss_function": loss_function.state_dict(),
             "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "epoch": epoch, "step": step, "args": vars(args)},
            temporary,
        )
        temporary.replace(checkpoint_file)  # a crash while saving never damages the last good checkpoint
        log_file = run_dir / "log.csv"
        new_log = not log_file.exists()
        with open(log_file, "a", newline="") as file:
            writer = csv.writer(file)
            if new_log:
                writer.writerow(["epoch", "loss", "training_accuracy", "lfw_accuracy", "minutes"])
            batches = totals["batches"]  # only every 200th step is measured; a tiny test set may have none
            writer.writerow([epoch + 1, round(totals["loss"] / batches, 3) if batches else "",
                             round(totals["accuracy"] / batches, 4) if batches else "",
                             round(lfw_accuracy, 4), round(minutes, 1)])

        hours_after_next_epoch = (time.time() - session_started + minutes * 60) / 3600
        if args.max_hours and hours_after_next_epoch > args.max_hours and epoch + 1 < args.epochs:
            print(f"Stopping after epoch {epoch + 1}: another epoch would pass the {args.max_hours} hour limit.")
            break

    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"Training finished. Network saved to {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
