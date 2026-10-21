#!/usr/bin/env python3
"""Deterministic synthetic test images for the parity checks (B4) — no photograph, nothing to license.

    python3 synthetic-set.py <out-dir>

Three PNGs, byte-identical on every run, with aspect ratios on both sides of square so the letterbox
pads top/bottom as well as left/right, plus one odd size that does not divide the model strides:

    noise-640x480.png     seeded uniform RGB noise
    gradient-480x640.png  smooth two-axis color gradient
    checker-333x517.png   64 px checkerboard at an odd size

They carry no detectable COCO objects — their golden result is "no detections", and their value is
numeric: every backend (eager, ONNX Runtime, IREE host and device) must produce the same head on them.
Needs `pip install numpy pillow`.
"""
import sys

import numpy as np
from PIL import Image


def main(out: str) -> None:
    rng = np.random.default_rng(20261021)
    noise = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    y, x = np.mgrid[0:640, 0:480].astype(np.float32)
    gradient = np.stack([x / 479 * 255, y / 639 * 255, (x / 479 + y / 639) / 2 * 255], axis=-1).astype(np.uint8)

    yy, xx = np.mgrid[0:517, 0:333]
    cells = ((yy // 64 + xx // 64) % 2).astype(np.uint8)
    checker = np.stack([cells * 255, cells * 200 + 30, (1 - cells) * 255], axis=-1).astype(np.uint8)

    for name, pixels in (("noise-640x480", noise), ("gradient-480x640", gradient), ("checker-333x517", checker)):
        path = f"{out}/{name}.png"
        Image.fromarray(pixels, "RGB").save(path, optimize=True)
        print(f"[synthetic-set] {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
