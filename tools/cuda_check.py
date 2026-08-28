"""Print what CUDA and TensorRT this process actually sees.

Run this when inference fails inside the kernels rather than at load time, e.g.
"Cask (Cask convolution execution)": that points at the runtime environment or
the engine, not at how the game calls TensorRT.

    python tools/cuda_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from rps.cuda import CudaMemory  # noqa: E402


def main() -> None:
    try:
        import tensorrt as trt
        print(f"tensorrt {trt.__version__} ({trt.__file__})")
    except ImportError as exc:
        print(f"tensorrt missing: {exc}")

    mem = CudaMemory()
    print(f"cuda backend: {mem._backend}")

    if mem._backend == "cudart":
        cudart = mem._cudart
        print("runtime version:", mem._check(cudart.cudaRuntimeGetVersion()))
        print("driver version :", mem._check(cudart.cudaDriverGetVersion()))
        count = mem._check(cudart.cudaGetDeviceCount())
        print("device count   :", count)
        for i in range(count):
            props = mem._check(cudart.cudaGetDeviceProperties(i))
            name = props.name.decode() if isinstance(props.name, bytes) else props.name
            print(f"  [{i}] {name} sm_{props.major}{props.minor}")
        free, total = mem._check(cudart.cudaMemGetInfo())
        print(f"memory         : {free / 1e6:.0f} MB free / {total / 1e6:.0f} MB")
    else:
        import pycuda.driver as drv
        print("driver version :", drv.get_version())
        dev = drv.Device(0)
        print(f"  [0] {dev.name()} sm_{dev.compute_capability()[0]}{dev.compute_capability()[1]}")

    # A round trip through device memory: if this fails, nothing else can work.
    src = np.arange(16, dtype=np.float32)
    dst = np.zeros_like(src)
    ptr = mem.alloc(src.nbytes)
    mem.htod(ptr, src)
    mem.dtoh(dst, ptr)
    mem.sync()
    print("memcpy round trip:", "ok" if np.array_equal(src, dst) else "MISMATCH")


if __name__ == "__main__":
    main()
