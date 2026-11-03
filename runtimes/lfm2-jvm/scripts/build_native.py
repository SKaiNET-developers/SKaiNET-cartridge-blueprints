#!/usr/bin/env python3
"""Compile the real ARM64 bridge with a pinned, offline Docker compiler image."""
import argparse
import json
from pathlib import Path
import subprocess

DEFAULT_IMAGE = "skainet/iree-compiler:3.11.0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        parser.error("Use a fresh empty native build output directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run-started").open("x") as stream:
        stream.write("Native ARM64 compilation; preserve this directory if the build fails.\n")
    inspected = subprocess.run(["docker", "image", "inspect", args.image], check=True,
                               capture_output=True, text=True, timeout=30)
    image, = json.loads(inspected.stdout)
    if image["Architecture"] != "arm64" or image["Os"] != "linux" or "IREE_VERSION=3.11.0" not in image["Config"]["Env"]:
        raise ValueError("Compiler image identity or platform mismatch")
    command = ["docker", "run", "--rm", "--network", "none", "--platform", "linux/arm64",
               "--entrypoint", "python3", "-v", str(args.source.resolve(strict=True)) + ":/source:ro",
               "-v", str(args.descriptors.resolve(strict=True)) + ":/descriptors:ro",
               "-v", str(output) + ":/out", image["Id"],
               "/source/build.py", "--descriptor", "/descriptors/descriptor.canonical.json",
               "--pack-descriptor", "/descriptors/descriptor.json", "--output", "/out/liblfm2_cartridge.so",
               "--cxx", "clang++"]
    with (output / "compiler.log").open("xb") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=240)
    record = {"schema_version": 1, "image_id": image["Id"], "requested_image": args.image, "platform": "linux/arm64", "network": "none",
              "image_repo_digests": image.get("RepoDigests", []), "command": command,
              "scope": "Native bridge compiled in the pinned image; no IREE or accelerator inference."}
    with (output / "compiler-container.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "built", "output": str(output)}))


if __name__ == "__main__":
    main()
