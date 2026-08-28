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
| `rps/logic.py` | left/right assignment and win/lose/draw rules |
| `rps/app.py` | pygame loop and rendering |
| `models/rps_yolo11n.onnx` | trained detector, 320x320 — kept only as the source for building an engine |

The engine's own class list is unreliable, so ids are mapped in `CLASS_NAMES`:
`{0: scissors, 1: rock, 2: paper}`, an order that is easy to get wrong.

## Jetson

Inference runs on a TensorRT engine through ultralytics, which handles
preprocessing and NMS. An engine is tied to the GPU and TensorRT version it was
built with, so build it on the board:

```bash
yolo export model=models/rps_yolo11n.onnx format=engine half=True imgsz=320
```

Then point `--model` at it (or copy it to `models/rps_yolo11n_2.engine`, the
default):

```bash
python main.py --model /home/aidl/work/rps_yolo11n_2.engine
```
