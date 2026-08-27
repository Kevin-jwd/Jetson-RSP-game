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
| `rps/detector.py` | YOLO11 inference (letterbox, decode, NMS) behind `detect(frame) -> [Detection]` |
| `rps/logic.py` | left/right assignment and win/lose/draw rules |
| `rps/app.py` | pygame loop and rendering |
| `models/rps_yolo11n.onnx` | trained detector, 320x320, classes `scissors, rock, paper` |

Class names are read from the ONNX metadata rather than hardcoded — this model
is ordered `scissors, rock, paper`, which is easy to get wrong.

## Jetson

`rps/app.py` only depends on the `detect()` interface, so deploying to the Orin
Nano means adding a TensorRT backend next to `OnnxDetector`; the game and UI are
unchanged. The engine must be built on the board:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/rps_yolo11n.onnx --fp16 --saveEngine=models/rps.engine
```
