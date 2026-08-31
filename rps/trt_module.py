"""Device memory for the TensorRT backend.

Everything runs on the default stream with synchronous copies: one frame at a
time is all the game needs, and a stream handle that TensorRT cannot execute on
is a hard failure to debug. JetPack images ship either cuda-python or pycuda
depending on the release, so use whichever is installed.
"""

from __future__ import annotations

import numpy as np


class CudaMemory:
    """Allocate device buffers and copy frames in and results out."""

    stream = 0  # default stream, which TensorRT accepts as a null handle

    def __init__(self):
        self._backend = None
        try:
            try:
                from cuda.bindings import runtime as cudart  # cuda-python >= 12.3
            except ImportError:
                from cuda import cudart  # cuda-python < 12.3
            self._cudart = cudart
            self._backend = "cudart"
        except ImportError:
            pass

        if self._backend is None:
            try:
                import pycuda.autoinit  # noqa: F401  (creates the context)
                import pycuda.driver as drv
            except ImportError as exc:
                raise RuntimeError(
                    "TensorRT backend needs cuda-python or pycuda (pip install cuda-python)"
                ) from exc
            self._drv = drv
            self._backend = "pycuda"

        self._keep = []  # pycuda frees its allocations when they go out of scope

    def _check(self, result):
        """cuda-python returns (error, value); raise on error, unwrap otherwise."""
        err, *rest = result if isinstance(result, tuple) else (result,)
        if err != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA call failed: {err}")
        return rest[0] if rest else None

    def alloc(self, nbytes: int) -> int:
        if self._backend == "cudart":
            return self._check(self._cudart.cudaMalloc(nbytes))
        buf = self._drv.mem_alloc(nbytes)
        self._keep.append(buf)
        return int(buf)

    def htod(self, dst: int, src: np.ndarray) -> None:
        if self._backend == "cudart":
            self._check(self._cudart.cudaMemcpy(
                dst, src.ctypes.data, src.nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ))
        else:
            self._drv.memcpy_htod(dst, src)

    def dtoh(self, dst: np.ndarray, src: int) -> None:
        if self._backend == "cudart":
            self._check(self._cudart.cudaMemcpy(
                dst.ctypes.data, src, dst.nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ))
        else:
            self._drv.memcpy_dtoh(dst, src)

    def sync(self) -> None:
        if self._backend == "cudart":
            self._check(self._cudart.cudaDeviceSynchronize())
        else:
            self._drv.Context.synchronize()
