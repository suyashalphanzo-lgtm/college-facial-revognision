# Build log: how we built our own SFace

This is the full story of how we built and trained our own face recognition model, a copy of the SFace model that comes with OpenCV. It runs from the first idea to the model the app uses now, in the order things happened, including what went wrong and how it was fixed. For the exact settings and every measurement, see the "Our own SFace" part of [CLAUDE.md](CLAUDE.md). How to run the app is in [README.md](README.md).

## At a glance

| | |
| --- | --- |
| What we built | A network that turns a photo of a face into 128 numbers, doing the same job as OpenCV's SFace |
| Network | MobileFaceNet: 1,003,264 weights, a 4 MB file (`models/our_sface.onnx`) |
| How it learns | The SFace loss |
| Training photos | CASIA-WebFace: 490,623 photos of 10,572 people |
| Test photos | LFW: 13,233 photos of 5,749 people |
| Where it trained | Kaggle's free GPU (one NVIDIA T4) |
| Training time | First model: 30 epochs, 3.2 hours. Second model: 90 epochs, 10.7 hours |
| Result | 99.13% on the standard LFW test (OpenCV's SFace: 99.35%) |
| In the app | The only face recognizer the app uses (OpenCV's YuNet still finds the faces) |
| When | 13–14 November 2026 |

## 1. The goal

The app already worked with two ready-made models from OpenCV: YuNet finds the faces in a picture, and SFace turns each face into 128 numbers that say whose face it is. We wanted to know whether we could make these ourselves. Building a face detector like YuNet is a bigger project, so we kept YuNet and rebuilt SFace, the part that recognizes people. The goals were:

- to learn how a face recognizer is trained,
- to get as close as we could to OpenCV's SFace,
- to plug our model into the app, so the two can be compared on the same faces.

## 2. How a face recognizer learns

The network never sees "is this the same person?" during training. It gets a simpler task: **"which of these 10,572 people is this?"** Every person in the training photos gets a *class centre*, a learnable list of 128 numbers. The network's 128 numbers for a face and all the centres are scaled to length 1, so they all lie on a sphere, and the angle between two of them says how alike they are. Training moves each face's numbers towards its own person's centre and away from everyone else's.

After training, the class centres are thrown away and only the network is kept. It has learned to put photos of the same person close together, and it still does that for people it never saw. That's why a network trained on celebrities can recognize college students.

OpenCV's description of its SFace models says they are "MobileFaceNet instances trained on the SFace loss function", so we needed those two pieces.

**The network: MobileFaceNet** (`training/mobilefacenet.py`). A small network designed for phones (Chen et al., 2018). It takes an aligned 112×112 face and shrinks it step by step (112 → 56 → 28 → 14 → 7 → 1 pixel) into 512 numbers, then 128. It has 1,003,264 weights. We wrote it out from the official SFace code (github.com/zhongyy/SFace), with explanations.

**The loss: SFace** (`training/sface_loss.py`, from Zhong et al., 2021). The loss is the number training tries to make smaller. It pulls each face towards its own centre and pushes it away from the others. SFace's idea is to weight the pull and the push with smooth S-shaped curves (sigmoids) of the angle, so both fade out once a face is good enough:

- the pull stops once the face is within about 50° of its own centre (a = 0.87 radians),
- the push stops once another person's centre is more than about 69° away (b = 1.2 radians).

Older losses keep pulling and pushing forever. Web photos contain mistakes (wrong names, blurry faces), and a network that keeps pushing on them ends up memorizing the mistakes. The other two settings are s = 64 (a scale) and k = 80 (how steep the curves are), all from the official code.

## 3. The data

- **Training: CASIA-WebFace,** 490,623 web photos of 10,572 celebrities, already aligned to 112×112 (the Kaggle dataset `yakhyokhuja/webface-112x112`, 2.72 GB). It is the dataset of the official SFace code's CASIA-WebFace experiments.
- **Testing: LFW** ("Labeled Faces in the Wild"), 13,233 news photos of 5,749 people. We used the copy scikit-learn downloads (181 MB), checked against its published checksums.
- **Both are for research use only.** They are not in this repository, and our model is for learning and research only.

`training/datasets.py` prepares them:

- **Packing:** opening 490,623 small files during training is slow, so they are packed into one 2.7 GB file plus an index saying where each photo starts. Training reads it through a memory map.
- **Aligning LFW:** every LFW photo is aligned by the app's own code (YuNet, and at first OpenCV's `alignCrop`; see section 17 for our own alignment), so the tests see faces exactly as the app does.

## 4. Two checks before training

1. **Are the training photos aligned like the app's faces?** OpenCV's `alignCrop` moves the eyes, nose and mouth corners onto a standard 5-point template. On 1,000 training photos, YuNet found the eyes and nose within 1–2 pixels of that template, but the mouth corners 4–5 pixels higher. LFW faces aligned by the app showed the same thing, so this is just where YuNet puts mouth corners. Training photos and the app's faces match within about 1 pixel.
2. **What exactly does OpenCV give a model?** We exported an untrained network and compared OpenCV's output with PyTorch's. OpenCV's `FaceRecognizerSF` gives the model raw RGB pixels from 0 to 255. So our network scales the pixels itself (`(pixel - 127.5) / 128`), and its file can replace OpenCV's SFace file directly.

`tests/test_our_sface.py` checks the network's input and output sizes and the loss.

## 5. The training loop

`training/train.py` uses the official SFace settings: plain SGD with momentum 0.9, a learning rate of 0.1 divided by 10 three times, weight decay 5e-4, batches of 256 photos, and each photo flipped left-right half the time. The *learning rate* is how big a step the network takes each time it learns from a batch: big steps learn fast but can't settle on fine details, and small steps settle.

We added a few things for a small GPU budget:

- a warm-up: the learning rate starts near 0 and rises to 0.1 over the first 1,000 steps,
- 16-bit arithmetic, which is faster (float16 with a gradient scaler on Kaggle's T4),
- no weight decay on batch-norm values and PReLU slopes,
- an LFW test after every *epoch* (one pass over all the training photos), so you can watch it improve,
- progress saved after every epoch, so a stopped run carries on where it left off.

## 6. The laptop was too hot

The plan was to train on this laptop's RTX 4050 GPU. It was already at 81°C before training. In a speed test it reached 89°C and slowed itself down to 210 MHz to cool off, a small fraction of its normal speed. It managed about 90 photos per second, so 30 epochs would have taken about two days. We moved to **Kaggle**, which gives free GPU time (30 hours a week, up to 12 hours per run).

## 7. Getting it to run on Kaggle

The Kaggle notebook (`training/our_sface_kaggle.ipynb`) is built from this project's own files by `training/make_kaggle_notebook.py`, so Kaggle runs exactly the code in this repository. Before uploading, we rehearsed the whole notebook on the laptop with the 12 people who have the most photos (7,866 photos). Two short epochs already lifted LFW accuracy from guessing to 67% and then 71%.

Kaggle still found four problems:

1. **The first run failed after 29 seconds.** A notebook cell that writes a file (`%%writefile`) refuses to write an empty one, and `app/__init__.py` is empty. Fix: empty files are made with `open(name, "w").close()`.
2. **Reading the dataset was far too slow.** On Kaggle, the dataset is a folder per person, and each file takes a while to arrive over the network: packing didn't reach 50,000 photos in 8 minutes. Downloading the zip inside Kaggle failed too (the link returns "404 not found" there). Fix: reading 64 files at once. Listing the folders then took 8 seconds, and packing ran at about 100,000 photos every 4 minutes.
3. **Two GPUs were 5.6 times slower than one.** Kaggle offers two T4 GPUs, and the first full run used both. It trained only 266 photos per second, so 30 epochs would have taken about 15 hours. A separate speed test measured photos per second: one T4 1,604 (with a faster memory layout called channels_last), one T4 without it 1,121, both T4s 286. For a network this small, copying it to the second GPU and collecting the results every step costs far more than the second GPU adds. Fix: train on one T4.
4. **A failed command doesn't stop a notebook.** Jupyter just moves on to the next cell, so a broken step could end as a "successful" run with no model. Fix: every step runs through a small `run()` function that stops the notebook when a step fails.

## 8. The first model (13 November 2026)

Version 6 of the notebook trained for 30 epochs, dropping the learning rate at epochs 16, 24 and 28. It ran at about 1,400 photos per second (5.8 minutes per epoch), 3.2 hours in all.

| Epoch | Learning rate | LFW accuracy (Kaggle's crops) |
| --- | --- | --- |
| 1 | 0.1 | 86.52% |
| 4 | 0.1 | 94.43% |
| 9 | 0.1 | 96.25% |
| 16 | 0.1 | 96.47% |
| 17 | 0.01 | 98.25% |
| 21 | 0.01 | 98.75% |
| 25 | 0.001 | 98.98% |
| 30 | 0.0001 | 98.95% |

**Lesson: the learning-rate drops did most of the late improvement.** At 0.1, accuracy climbed fast and then stalled around 96–96.6%. The first drop added almost 2 points in one epoch, the second a little, the third nothing. The full log is `training/runs/mobilefacenet_sface/log.csv`.

## 9. Exporting and testing

**Export** (`training/export_onnx.py`): the trained network is saved as `models/our_sface.onnx`, a 4 MB file that OpenCV loads with `cv2.FaceRecognizerSF.create`, like the original SFace. (The app later switched to OpenCV's general `cv2.dnn` engine to run it; see section 17.) PyTorch's newer exporter crashed on the Windows console (it prints a ✅ symbol the console can't show), so we use the older one. OpenCV's output equals PyTorch's (similarity 1.000000). Files downloaded from Kaggle are loaded with `weights_only=True`, so they can only contain numbers, never hidden code.

**98.95% on Kaggle, 98.82% here.** Same network, different LFW crops: Kaggle aligned LFW with OpenCV 4.14 and this computer with OpenCV 5.0, and the two place faces slightly differently. So every comparison uses this computer's crops.

**Two tests** (`training/evaluate.py`):

- **The standard LFW test:** 6,000 pairs of photos, half of them the same person. Research papers report this number.
- **The college test:** like the app's real use. 100 people are registered with 5 photos each, up to 5 other photos of each should get the right name, and 1,000 strangers should be "Unknown". It is averaged over 3 random draws of people. Each model's match threshold is chosen with one rule: the lowest threshold at which at most 0.4% of strangers get a name.

**First model against OpenCV's SFace:**

| | Standard LFW test | Threshold | Named correctly, 5 photos | Named correctly, 1 photo |
| --- | --- | --- | --- | --- |
| OpenCV's SFace | 99.35% | 0.45 | 99.6% | 95.2% |
| Our first model | 98.82% | 0.49 | 95.9% | 68.1% |

(The app uses 0.46 for OpenCV's SFace, a value chosen earlier with the same rule on other random draws. At 0.46 it names 99.4%.)

Ours needed a *higher* threshold than OpenCV's in the college test, even though it needed a lower one for single pairs (0.25 against 0.31). A few strangers get unusually high similarities to someone in the class, so the threshold has to sit higher, and there it also cuts off more of the registered people.

## 10. Plugging it into the app

One line in `app/config.py` chooses the model: `RECOGNIZER = "sface"` (OpenCV's) or `"ours"`. The two models' numbers mean different things, so each has its own database file and its own threshold. The first time a model is used, the app registers everyone again from the photo folders. If `models/our_sface.onnx` is missing, the app stops with a message saying how to make it.

Later the lab's rule, that the project must use its own model, made ours the only one: see section 17.

## 11. Ideas to make it better

1. **The mirror trick.** Also run the mirror image of each face through the model and average the two results. Tested straight away: it lifted our first model from 98.82% to 98.95% on the standard test, and from 95.9% to 96.7% in the college test (threshold 0.48). It did nothing for OpenCV's SFace. So the app uses it with our model only. It costs about 3 ms more per face.
2. **Train longer.** The official recipe trains 125 epochs and drops the learning rate at 50, 70 and 80. Ours had trained only 30. We chose 90 epochs: after the last drop, the learning rate is too small to change much (our first model's last drop gained nothing), and 90 epochs fit inside one 12-hour Kaggle run.
3. **Webcam-style training photos.** The training photos are mostly sharp, well-lit celebrity photos, but webcams give small, blurry, dark or washed-out faces. So each training photo is randomly changed (`webcam_style` in `training/datasets.py`): 20% are shrunk to 24–80 pixels and enlarged back, 20% are blurred, and 30% get different brightness and contrast. About 45% stay unchanged.
4. **A bigger dataset or a bigger network: not done.** Datasets with millions of photos would take over an hour per epoch on Kaggle, and several were withdrawn over privacy concerns. A bigger network would be slower in the app, and OpenCV's SFace uses the same small network as ours.

Ideas 2 and 3 went into one run.

## 12. The second model (14 November 2026)

Version 7 of the notebook trained for 90 epochs, with drops at 50, 70 and 80 and the webcam-style photo changes. It took 38,408 seconds (10.7 hours), and used about a third of the week's free GPU time.

**It trained more slowly than expected:** about 1,190 photos per second instead of 1,400 (6.8 minutes per epoch instead of 5.8). On the laptop, the photo changes added only about 0.1 ms per photo. Kaggle's processors prepare photos more slowly, though, and with the extra work, preparing the photos probably became the bottleneck there. A speed test on Kaggle itself would have shown this before the run.

| Epoch | Learning rate | LFW accuracy (Kaggle's crops) |
| --- | --- | --- |
| 1 | 0.1 | 83.82% |
| 9 | 0.1 | 96.17% |
| 50 | 0.1 | 97.25% |
| 51 | 0.01 | 98.67% |
| 70 | 0.01 | 98.78% |
| 71 | 0.001 | 99.12% |
| 81 | 0.0001 | 99.18% |
| 90 | 0.0001 | 99.13% |

The same pattern as the first model, stretched out: a long stall at about 97%, a big jump at the first drop, a smaller one at the second, nothing at the third. The full log is `training/runs/mobilefacenet_sface_v2/log.csv`.

(When downloading the results, Chrome saved `model.pt` and `last.pt` as `model.zip` and `last.zip`. A PyTorch `.pt` file is a zip archive inside; renaming them back is all that's needed.)

## 13. Final results

The college test with only 3 random draws made the second model look slightly worse than the first (96.3% against 96.7% named), but each draw has only about 350 test photos, so that's one or two photos. So we repeated it with 20 draws and finer thresholds (steps of 0.002). Each model is at its own threshold, with at most 0.4% of strangers named:

| Model | Standard LFW test | Threshold | Named, 5 photos | Named, 3 photos | Named, 1 photo |
| --- | --- | --- | --- | --- | --- |
| OpenCV's SFace | 99.35% | 0.454 | 99.5% | 99.3% | 95.6% |
| Our first model, mirror trick | 98.95% | 0.480 | 97.1% | 95.4% | 76.7% |
| **Our second model, mirror trick** | **99.13%** | 0.482 | 96.9% | 95.4% | 77.1% |

What this shows:

- **Training three times longer helped the standard test:** about 52 mistakes in 6,000 pairs instead of 71, a quarter fewer.
- **It did not help the college test:** both of our models name about 97% of registered people. The extra training fixed moderately hard pairs, but the strict college-test threshold is set by the hardest cases: strangers who look like someone in the class, and photos of a person that look unlike their other photos.
- **On 5 webcam photos of one person, the result was mixed.** The second model scored higher than the first on the two hardest photos (tilt up, tilt down) and lower on another, with the same average. So the webcam-style changes made no visible difference there. One person is only a hint.
- **The mirror trick no longer matters much:** with the second model, it only helps a little with 1 photo per person. The app keeps it on.

The app then used the second model when `RECOGNIZER = "ours"`, at threshold 0.48 with the mirror trick. In a class registered with 5 photos each, about 3 in 100 students would show up as "Unknown" with it, against about 1 in 200 with OpenCV's SFace. So at first OpenCV's SFace stayed the app's default, until the lab's rule made ours the only one (section 17).

## 14. Why ours is still behind

Our model uses the same network and the same loss as OpenCV's SFace. Training length was the one difference we knew of, and training three times longer didn't close the gap in the college test. The likeliest remaining difference is the training data. The official SFace code also supports VGGFace2 and MS1MV2, which are much bigger than CASIA-WebFace, but OpenCV doesn't say what data its model was trained on.

## 15. What we learned

- A face recognizer is trained as a classifier ("which person is this?"), and the classifier part is thrown away afterwards.
- How a model will be fed matters: the faces must be aligned the same way, and the pixels given in the same form, in training and in use.
- A small, cheap rehearsal run catches most mistakes before spending GPU hours.
- More GPUs isn't always faster: for a small network, one GPU was 5.6 times faster than two.
- With a learning rate that drops in steps, most of the late improvement comes right after each drop, and the last drop added nothing in both runs.
- Measure on the task you care about. Training longer improved the standard test but not the college test.
- Small differences need many repeats: with 3 random draws, a 0.4-point difference was luck.
- One-to-many recognition (one face against a whole class) needs a stricter threshold than comparing two photos.

## 16. How to build it again

Training needs PyTorch and onnx on top of the app's libraries:

```bash
pip install -r training/requirements.txt
```

1. Download the Kaggle dataset `yakhyokhuja/webface-112x112` and save its zip as `datasets/webface_112x112.zip`.
2. Pack the training photos, then download and align LFW:

   ```bash
   python -m training.datasets
   ```

3. Train. On a computer with a well-cooled GPU, run `python -m training.train`. Otherwise use Kaggle, following "Training on Kaggle" in [CLAUDE.md](CLAUDE.md).
4. Turn the trained network into the app's model file:

   ```bash
   python -m training.export_onnx
   ```

5. Test it on LFW (about a minute). With `--compare-opencv` it also tests OpenCV's SFace the same way, as a yardstick:

   ```bash
   python -m training.evaluate --compare-opencv
   ```

6. Set `MATCH_THRESHOLD` in `app/config.py` to the threshold the test chose. If `data/face_database_ours.json` exists, delete it, so everyone is registered again with the new model.

## 17. Only our own model in the app (15 November 2026)

The lab requires the project to use its own model, so OpenCV's SFace was taken out of the app. Our SFace is now the only recognizer: the `RECOGNIZER` switch is gone, the app never downloads OpenCV's SFace, and it uses threshold 0.48 with the mirror trick. Two pieces of OpenCV's SFace code were still in use, so they were replaced with our own:

- **Alignment.** Turning and cutting out each face used OpenCV's `alignCrop`. It belongs to OpenCV's `FaceRecognizerSF` class, which can only be created with a model file, but on Kaggle the LFW photos are aligned before our model exists. So the calculation is now written out in `app/face_engine.py` (`align_face`). It is a least-squares fit (Umeyama's method) of the turn, scale and shift that move the 5 face points onto the standard template; then `cv2.warpAffine` cuts out the 112×112 face. On all 13,233 LFW photos it gave the same crops as `alignCrop`, apart from 1 brightness step in 0.03% of the pixels (OpenCV rounds some numbers differently), and both LFW tests gave the same results.
- **Running the model.** The app ran our network through `FaceRecognizerSF`. Now OpenCV's general deep-learning engine (`cv2.dnn`) runs it, with the same preparation: BGR colours turned into RGB. On 2,000 LFW faces the numbers were exactly the same.

OpenCV's SFace is still used in one place: `python -m training.evaluate --compare-opencv` downloads it to measure ours against it, which is how the comparisons in this log were made. In the app, only the face detector, YuNet, is still OpenCV's model.
