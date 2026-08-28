# Jetson RPS Game

Two-player rock-paper-scissors over a webcam. Detects both hands in one frame,
assigns them to the left and right player by position, and shows each side's
result.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Options: `--model`, `--camera`, `--conf`, `--no-mirror`. Keys: `m` mirror, `q` quit.

## Layout

| Path | Role |
| --- | --- |
| `rps/detector.py` | YOLO11 inference on a TensorRT engine behind `detect(frame) -> [Detection]` |
| `rps/cuda.py` | device memory and stream (cuda-python or pycuda) |
| `tools/probe.py` | headless check of what the engine returns |
| `rps/logic.py` | left/right assignment and win/lose/draw rules |
| `rps/app.py` | pygame loop and rendering |
| `models/rps_yolo11n.onnx` | trained detector, 320x320 — kept only as the source for building an engine |

Class names come from the engine's own JSON header when it has one; otherwise the
`CLASS_NAMES` fallback applies, `{0: scissors, 1: rock, 2: paper}` — an order that
is easy to get wrong.

## Jetson

Inference talks to the `tensorrt` module from JetPack directly, so neither torch
nor ultralytics is needed on the board — only `cuda-python` (or `pycuda`) for
device memory:

```bash
pip install cuda-python
```

An engine is tied to the GPU and TensorRT version it was built with, so build it
on the board:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/rps_yolo11n.onnx --fp16 --saveEngine=models/rps_yolo11n_2.engine
```

Then run, pointing `--model` at the engine if it lives elsewhere:

```bash
python main.py --model /home/aidl/work/rps_yolo11n_2.engine
```

If the game shows no boxes, `tools/probe.py` prints the engine's input/output
shapes and raw scores without a GUI.
