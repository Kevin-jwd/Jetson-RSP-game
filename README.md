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
| `rps/detector.py` | YOLO11 inference behind `detect(frame) -> [Detection]`; onnxruntime and TensorRT backends |
| `rps/cuda.py` | device memory for the TensorRT backend (cuda-python or pycuda) |
| `rps/logic.py` | left/right assignment and win/lose/draw rules |
| `rps/app.py` | pygame loop and rendering |
| `models/rps_yolo11n.onnx` | trained detector, 320x320, classes `scissors, rock, paper` |

Class names are read from the ONNX metadata rather than hardcoded — this model
is ordered `scissors, rock, paper`, which is easy to get wrong.

## Jetson

`--model` picks the backend by extension, so the same command runs either one:

```bash
python main.py --model /home/aidl/work/rps_yolo11n_2.engine
```

An engine is tied to the GPU and TensorRT version it was built with, so build it
on the board:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/rps_yolo11n.onnx --fp16 --saveEngine=models/rps.engine
```

Engines exported by ultralytics carry their class names in a JSON header and are
read back automatically. A `trtexec` engine has no such header, so the backend
falls back to this model's order, `scissors, rock, paper`.
