# Jetson RPS Game

Rock-paper-scissors over a webcam, running on a TensorRT engine on a Jetson Orin
Nano. The title screen offers two modes:

- **1인용** — one hand against a random move, drawn over the left of the video.
  Rounds keep going until 종료.
- **2인용** — two hands in the frame, judged left against right. One round, then
  재시도 or 종료.

The game chants 가위-바위-보, watches for 0.7s and judges by majority vote, then
freezes that frame with the verdict. Only `q` closes the program.

## Run

```bash
pip install -r requirements.txt
python main.py --model models/best.engine
```

Options: `--model`, `--camera`, `--conf`, `--classes`, `--no-mirror`.
Keys: `r` retry, `m` mirror, `q` quit (the only way out). Labels fall back to
English when no Hangul font is installed.

## Layout

| Path | Role |
| --- | --- |
| `main.py` | argument parsing |
| `rps/detector.py` | YOLO11 inference on a TensorRT engine behind `detect(frame) -> [Detection]` |
| `rps/cuda.py` | device memory (cuda-python or pycuda) |
| `rps/logic.py` | win/lose/draw rules |
| `rps/app.py` | round state machine (menu → countdown → shoot → result), buttons, rendering |
| `rps/particles.py` | particle burst over the winning hand |
| `assets/` | `rock/paper/scissors` images for the AI's hand (optional) |
| `tools/probe.py` | headless check of what the engine returns |
| `tools/cuda_check.py` | CUDA and TensorRT environment check |
| `tools/merge_dataset.py` | merge YOLO datasets, remapping class ids by name |
| `models/rps_yolo11n.onnx` | first detector, 320x320 — kept as a source for building an engine |

`docs/ARCHITECTURE.md` covers how each module works and why.

## Judging

A single frame is a bad witness. The chant ends while the hand is still moving,
and a fist opening into paper passes through something the model reads as
scissors. So the game collects labels over `VOTE_MS` (0.7s), one vote per frame
per hand, and takes the majority; ties go to the higher summed confidence. Hands
are matched to players by box position, not by label.

## Class order

Class order is easy to get wrong and fails silently — rock read as paper looks
like a badly trained model rather than a bug. Only indices 0 and 2 differ between
the common orderings, so **rock stays correct while paper and scissors swap**. If
you see that, suspect the order, not the accuracy.

The order is taken from `--classes` first, then the engine's own JSON header if
it has one (ultralytics exports), then the `CLASS_NAMES` fallback in
`rps/detector.py`, which follows the Roboflow dataset: `paper, rock, scissors`.
An engine built from the older class dataset needs `--classes scissors,rock,paper`.

## Jetson

Inference talks to the `tensorrt` module from JetPack directly, so neither torch
nor ultralytics is needed on the board — only `cuda-python` (or `pycuda`) for
device memory:

```bash
pip install cuda-python
```

An engine is tied to the GPU and TensorRT version it was built with, so build it
on the board. Give each build its own name; overwriting the engine you are
currently playing with leaves nothing to fall back to:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=models/best.engine --fp16
```

When nothing is detected, check the engine before the game — `tools/probe.py`
prints its input/output shapes, its class names, and the raw scores at a very low
threshold, with no GUI:

```bash
python tools/probe.py --model models/best.engine
```

`tools/cuda_check.py` prints the TensorRT and CUDA versions the process actually
sees, which separates an environment problem from a code one.
