#!/usr/bin/env python3
"""The ONNX Runtime half of the B4 host parity check (the eager half is `parityEager` / YoloV8ParityCli).

Runs the Ultralytics ONNX export on the letterboxed input the CLI wrote and stores the raw head next to it,
where the CLI's second run picks it up:

    python3 ort-head.py <yolov8n.onnx> <work-dir>      # <work>/input.f32 -> <work>/ort-head.f32

Files are raw little-endian float32, no header; the input is [1, 3, S, S], the head [1, 4 + C, A].
Needs `pip install onnxruntime numpy`.
"""
import sys

import numpy as np
import onnxruntime as ort


def main(model: str, work: str) -> None:
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    (inp,) = session.get_inputs()
    data = np.fromfile(f"{work}/input.f32", dtype=np.float32).reshape([d for d in inp.shape])
    (head,) = session.run(None, {inp.name: data})
    head.astype(np.float32).tofile(f"{work}/ort-head.f32")
    print(f"[ort-head] {inp.name}{list(data.shape)} -> {list(head.shape)} ({ort.__version__}) -> {work}/ort-head.f32")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
