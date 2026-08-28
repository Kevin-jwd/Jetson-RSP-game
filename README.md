# Jetson RPS Game

Rock-paper-scissors over a webcam. The title screen offers `vs AI` (one hand
against a random move, drawn over the left of the video) and `vs Person` (two
hands in the frame). The game chants 가위-바위-보, reads the hands on the beat,
freezes that frame with the verdict, and stops there: 재시도 or 종료.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Options: `--model`, `--camera`, `--conf`, `--classes`, `--no-mirror`.
Keys: `r` retry, `m` mirror, `q` quit. Labels fall back to English when no
Hangul font is installed.

## Layout

| Path | Role |
| --- | --- |
| `rps/detector.py` | YOLO11 inference on a TensorRT engine behind `detect(frame) -> [Detection]` |
| `rps/cuda.py` | device memory and stream (cuda-python or pycuda) |
| `tools/probe.py` | headless check of what the engine returns |
| `rps/logic.py` | left/right assignment and win/lose/draw rules |
| `rps/app.py` | round state machine (idle → countdown → shoot → result), buttons, rendering |
| `rps/particles.py` | particle burst over the winning hand |
| `assets/` | `rock/paper/scissors` images for the AI's hand (optional) |
| `models/rps_yolo11n.onnx` | trained detector, 320x320 — kept only as the source for building an engine |

Class order is easy to get wrong and fails silently — rock read as paper looks like
a badly trained model. `--classes` wins, then the engine's own JSON header if it
has one, then the `CLASS_NAMES` fallback in `rps/detector.py`, which follows the
Roboflow dataset: `paper, rock, scissors`. An engine built from the older class
dataset needs `--classes scissors,rock,paper`.

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
