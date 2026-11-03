#!/usr/bin/env python3
"""Build the source-only C ABI evaluation bridge; does not load a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(items):
    value = {}
    for name, child in items:
        if name in value:
            raise ValueError("Descriptor contains a duplicate JSON key: " + name)
        value[name] = child
    return value


def canonical_descriptor(data):
    parsed = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
    if not isinstance(parsed, dict):
        raise ValueError("Descriptor must be a JSON object")
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--pack-descriptor", type=Path,
                        help="Exact descriptor.json bytes the materializer will place in the pack; defaults to --descriptor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cxx", default="clang++")
    parser.add_argument("--static-libstdcxx", action="store_true")
    args = parser.parse_args()
    output = args.output.absolute()
    report_path = output.with_name(output.name + ".build.json")
    if any(path.exists() or path.is_symlink() for path in (output, report_path)):
        parser.error("Use new library and build-report paths")
    descriptor = args.descriptor.read_bytes()
    try:
        canonical = canonical_descriptor(descriptor)
    except (ValueError, UnicodeError) as error:
        parser.error(str(error))
    if descriptor != canonical or len(descriptor) > 4 * 1024 * 1024 or b"\0" in descriptor:
        parser.error("Descriptor must use canonical compact sorted UTF-8 JSON, without a newline")
    pack_descriptor_path = args.pack_descriptor if args.pack_descriptor is not None else args.descriptor
    pack_descriptor = pack_descriptor_path.read_bytes() if args.pack_descriptor is not None else descriptor
    if len(pack_descriptor) > 4 * 1024 * 1024 or b"\0" in pack_descriptor:
        parser.error("Pack descriptor must be at most 4 MiB and contain no NUL bytes")
    try:
        if canonical_descriptor(pack_descriptor) != descriptor:
            parser.error("Pack descriptor and canonical descriptor identify different JSON objects")
    except (ValueError, UnicodeError) as error:
        parser.error(str(error))
    system = platform.system()
    if system not in ("Darwin", "Linux"):
        parser.error("Only Darwin and Linux builds are supported")
    if args.static_libstdcxx and system != "Linux":
        parser.error("Static libstdc++ is supported only for Linux")
    compiler = shutil.which(args.cxx)
    if compiler is None:
        parser.error("C++ compiler not found")
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True, text=True, timeout=20).stdout
    sources = [ROOT / "bridge.cpp", ROOT / "include/cartridge_abi.h", ROOT / "LICENSE.upstream", ROOT / "upstream.json", Path(__file__).resolve()]
    snapshots = {str(path.relative_to(ROOT)): path.read_bytes() for path in sources}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lfm2-native-build-", dir=output.parent) as temporary:
        work = Path(temporary)
        for name, data in snapshots.items():
            snapshot = work / "sources" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(data)
        header = work / "embedded_descriptor.h"
        header.write_text("static const unsigned char kDescriptorBytes[] = {" +
                          ",".join(str(value) for value in descriptor + b"\0") + "};\n" +
                          "static const unsigned char kPackDescriptorBytes[] = {" +
                          ",".join(str(value) for value in pack_descriptor + b"\0") + "};\n")
        product = work / output.name
        command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-pthread",
                   "-dynamiclib" if system == "Darwin" else "-shared",
                   "-I", str(work / "sources/include"), "-I", str(work), str(work / "sources/bridge.cpp"), "-o", str(product)]
        if system == "Darwin":
            command += ["-Wl,-install_name,@rpath/" + output.name]
        if args.static_libstdcxx:
            command += ["-static-libstdc++", "-static-libgcc"]
        subprocess.run(command, check=True, timeout=180)
        report = {
            "status": "built", "scope": "native_abi_child_jvm_evaluation_bridge",
            "descriptor_sha256": hashlib.sha256(descriptor).hexdigest(),
            "pack_descriptor_sha256": hashlib.sha256(pack_descriptor).hexdigest(),
            "descriptor_input": {"file": str(args.descriptor.absolute()),
                                 "sha256": hashlib.sha256(descriptor).hexdigest(), "size_bytes": len(descriptor)},
            "pack_descriptor_input": {"file": str(pack_descriptor_path.absolute()),
                                      "sha256": hashlib.sha256(pack_descriptor).hexdigest(),
                                      "size_bytes": len(pack_descriptor)},
            "output": {"file": str(output), "sha256": sha(product), "size_bytes": product.stat().st_size},
            "compiler": {"executable": compiler, "version": version.strip(), "argv": command},
            "platform": {"system": system, "machine": platform.machine()},
            "sources": [{"file": name, "sha256": hashlib.sha256(data).hexdigest()} for name, data in snapshots.items()],
            "source_snapshot": "Source/header bytes captured before compilation; compiler reads only those snapshots",
            "static_libstdcxx": args.static_libstdcxx,
            "model_execution": False, "full_conformance": False, "sealed_cartridge": False,
        }
        temporary_report = work / report_path.name
        temporary_report.write_text(json.dumps(report, indent=2) + "\n")
        # Link publication is exclusive, even if another process creates an output
        # after the initial existence check. Both temporary files share its filesystem.
        os.link(product, output)
        os.link(temporary_report, report_path)
    print(json.dumps({"status": "built", "library": str(output), "report": str(report_path)}))


if __name__ == "__main__":
    main()
