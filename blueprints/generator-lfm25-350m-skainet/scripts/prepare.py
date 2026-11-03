#!/usr/bin/env python3
"""Stage exact native/JVM execution bytes and gate unsigned blueprint materialization.

Self-contained on the device; no checkout, downloads, Java bundling, acceptance or
signing is needed. A staged directory is explicitly unmeasured until the separate
benchmark validator accepts evidence. The descriptor itself is never rewritten.
"""
from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile

REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"
MODEL_LICENSE = "LicenseRef-LFM-Open-License-1.0"
MODEL_FILES = {
    "model.safetensors": (708984464, "1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e"),
    "config.json": (1281, "720b43d6ddc2ed25be23eed355aefcf342434a176dedad23dbe0a5e3ac24bbb8"),
    "generation_config.json": (134, "0ec7d228c4de7e3453e98bf6910c7eb7841e87fede63ea11c0f75a5c64e1c3c3"),
    "tokenizer_config.json": (595, "3701f370c70034d06e28947ff51c1063983393319f429e3ce58f06a45fae67cc"),
    "chat_template.jinja": (5487, "ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6"),
    "tokenizer.json": (4733389, "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd"),
    "LICENSE": (10574, "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c"),
}
PERFORMANCE = {
    "measured_on": "Linux AArch64 CPU, external Java 21",
    "methodology": "artifacts/evaluation/native-abi-workload.json",
}
QUALITY = {"unmeasured": True,
           "harness": "Task-quality evaluation remains pending; exact HF fidelity is not task quality."}
RUNTIME = "artifacts/runtime/liblfm2_cartridge.so"
CANONICAL = ".provenance/descriptor.canonical.json"
REPORT_PATHS = {
    "native-benchmark": "artifacts/evaluation/native-abi-benchmark.json",
    "native-measurement-validation": "artifacts/evaluation/native-abi-validation.json",
    "benchmark-supervisor": "artifacts/evaluation/native-abi-supervisor.json",
}


def unique_object(items):
    value = {}
    for name, child in items:
        if name in value:
            raise ValueError("Duplicate JSON key: " + name)
        value[name] = child
    return value


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    return {"sha256": sha(path), "size_bytes": path.stat().st_size}


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def write_json(path: Path, value) -> None:
    write_new(path, (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii"))


def claim_directory(path: Path) -> None:
    # Gradle creates an empty @OutputDirectory before invoking Exec. Anything
    # already present, including the claim of a failed run, must survive.
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise FileExistsError(f"Use a fresh empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    write_new(path / ".run-started", b"unsigned evaluation staging; incomplete until execution.json exists\n")


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) != value:
        raise ValueError(f"Unsafe artifact path: {value}")
    return Path(*path.parts)


def verify(path: Path, expected: dict) -> None:
    actual = identity(path)
    if actual != {"sha256": expected["sha256"], "size_bytes": expected["size_bytes"]}:
        raise ValueError(f"Artifact identity mismatch: {path}")


def place(source: Path, destination: Path, expected: dict, *, hardlink: bool = False) -> None:
    verify(source, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    linked = False
    if hardlink:
        try:
            os.link(source, destination)
            linked = True
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
                raise
    if not linked:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    verify(destination, expected)


def verify_model(directory: Path) -> None:
    for name, (size, digest) in MODEL_FILES.items():
        verify(directory / name, {"sha256": digest, "size_bytes": size})


def worker_configuration() -> dict:
    return {"model_dir": "artifacts/model", "checkpoint_sha256": MODEL_FILES["model.safetensors"][1],
            "metadata_sha256": {name: value[1] for name, value in MODEL_FILES.items() if name != "model.safetensors"},
            "decoder_policy": "HUGGING_FACE_5_COMPAT", "default_max_output_tokens": 32, "max_output_tokens": 256}


def descriptor_pair(raw: Path, compact: Path) -> dict:
    value = read(raw)
    compact_bytes = compact.read_bytes()
    if canonical(value) != compact_bytes:
        raise ValueError("Canonical ABI descriptor differs from raw pack descriptor")
    # Native build.py uses UTF-8 JSON. This blueprint deliberately has ASCII
    # content, so both encoders must produce exactly the same bytes.
    utf8 = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")
    if utf8 != compact_bytes:
        raise ValueError("This blueprint requires ASCII descriptor strings")
    if value.get("performance") != PERFORMANCE or value.get("license") != "NOASSERTION":
        raise ValueError("Unexpected descriptor performance provenance or license")
    if value.get("target", {}).get("abi") != "aarch64-linux-gnu":
        raise ValueError("Only the measured ARM64 Linux target is supported")
    return value


def validate_build(library: Path, report: Path, raw: Path, compact: Path) -> None:
    value = read(report)
    if value.get("status") != "built" or value.get("platform") != {"system": "Linux", "machine": "aarch64"}:
        raise ValueError("Native build report does not identify a built Linux AArch64 library")
    if value.get("descriptor_sha256") != sha(compact) or value.get("pack_descriptor_sha256") != sha(raw):
        raise ValueError("Build report descriptor identity differs from this execution")
    verify(library, value["output"])
    if value.get("static_libstdcxx") is not False or not value.get("sources"):
        raise ValueError("Native library must retain source provenance and the evaluated dynamic C++ runtime linkage")


def closure(pack: Path) -> list[dict]:
    paths = [pack / "descriptor.json"] + sorted((pack / "artifacts").rglob("*"))
    records = []
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"No Java or other symlink may be an execution artifact: {path}")
        if path.is_dir():
            continue
        name = path.relative_to(pack).as_posix()
        records.append({"file": name, **identity(path), "source": name})
    return records


def role(path: str) -> str:
    return ("descriptor" if path == "descriptor.json" else "runtime" if path == RUNTIME else
            "weights" if path == "artifacts/model/model.safetensors" else
            "tokenizer" if path == "artifacts/model/tokenizer.json" else
            "calibration" if path.startswith("artifacts/evaluation/") else "other")


def finish_stage(pack: Path) -> None:
    descriptor = descriptor_pair(pack / "descriptor.json", pack / CANONICAL)
    verify_model(pack / "artifacts/model")
    if read(pack / "artifacts/runtime/worker.json") != worker_configuration():
        raise ValueError("Worker configuration is not the pinned configuration")
    validate_build(pack / RUNTIME, pack / "artifacts/runtime/native-build.json",
                   pack / "descriptor.json", pack / CANONICAL)
    artifacts = closure(pack)
    if sum(role(item["file"]) == "runtime" for item in artifacts) != 1:
        raise ValueError("Exactly one actual native runtime artifact is required")
    record = {"schema_version": 1, "status": "unmeasured_provisional_execution", "pack_files": artifacts,
              "descriptor_canonical": {"file": CANONICAL, **identity(pack / CANONICAL)},
              "external_java": {"bundled": False, "selection": "JAVA_HOME/bin/java", "major_version": 21,
                                "modules": ["jdk.incubator.vector"]},
              "scope": "Exact provisional execution bytes; performance acceptance and final materialization are separate gates."}
    write_json(pack / "execution.json", record)
    manifest = {"schema_version": "evaluation-execution-v1", "kind": "unsigned-evaluation-stage",
                "id": descriptor["id"], "version": descriptor["version"], "effective_license": "NOASSERTION",
                "descriptor_digest": "sha256:" + sha(pack / CANONICAL), "digest_alg": "sha256",
                "status": "unmeasured_provisional_execution", "signatures_present": False,
                "artifacts": [{"role": role(item["file"]), "path": item["file"],
                               "digest": "sha256:" + item["sha256"], "size": item["size_bytes"],
                               "license": MODEL_LICENSE if item["file"].startswith("artifacts/model/") else "NOASSERTION"}
                              for item in artifacts],
                "scope": "Staging metadata only; not a signed Cartridge Spec manifest."}
    write_json(pack / "manifest.json", manifest)


def profile(args) -> None:
    blueprint = read(args.blueprint)
    source, = blueprint["sources"]
    expected = {name: (size, digest) for name, (size, digest) in MODEL_FILES.items()}
    observed = {item["path"]: (item["size"], item["digest"].removeprefix("sha256:")) for item in source["files"]}
    if observed != expected or source["revision"] != REVISION or blueprint["license"] != "NOASSERTION":
        raise ValueError("Blueprint source/license pins changed")
    model = args.model_dir.resolve(strict=True)
    verify_model(model)
    value = {"spec_version": "0.4", "kind": "materialization-profile",
             "blueprint": {"id": blueprint["id"], "version": blueprint["version"], "digest": "sha256:" + sha(args.blueprint)},
             "target": "linux-aarch64", "cartridge": {"id": blueprint["id"] + "-linux-aarch64", "version": blueprint["version"]},
             "mirrors": {"liquid-lfm25-350m": model.as_uri() + "/"},
             "measurements": {"performance": PERFORMANCE, "quality": QUALITY}}
    write_json(args.output, value)
    write_json(args.output.with_name("preview-status.json"), {
        "status": "unmeasured_descriptor_preview", "materialization_authorized_by_this_file": False,
        "explanation": "Stable provenance fields only, used to compile exact descriptor bytes before measurement. The real materializer requires an independently accepted profile."})


def prepare(args) -> None:
    claim_directory(args.output)
    descriptor_pair(args.descriptor, args.canonical_descriptor)
    verify_model(args.model_dir)
    validate_build(args.native_library, args.native_build_report, args.descriptor, args.canonical_descriptor)
    mappings = {
        "descriptor.json": args.descriptor, CANONICAL: args.canonical_descriptor,
        RUNTIME: args.native_library, "artifacts/runtime/native-build.json": args.native_build_report,
        "artifacts/runtime/compiler-container.json": args.compiler_report,
        "artifacts/runtime/LICENSE.upstream": args.native_license,
        PERFORMANCE["methodology"]: args.workload,
    }
    jars = sorted(args.worker_libs.iterdir())
    if not jars or any(path.suffix != ".jar" or not path.is_file() or path.is_symlink() for path in jars):
        raise ValueError("Worker lib directory must contain only actual regular JAR products")
    if not any("sk/ainet/cartridge/generator/lfm2/worker/MainKt.class" in zipfile.ZipFile(path).namelist() for path in jars):
        raise ValueError("Actual native-worker main class is missing from installed JARs")
    mappings.update({"artifacts/jvm/lib/" + path.name: path for path in jars})
    mappings.update({"artifacts/model/" + name: args.model_dir / name for name in MODEL_FILES})
    for relative, source in mappings.items():
        # Build products must not share writable inodes with a later installDist
        # or compiler invocation. Only the verified immutable checkpoint is linked.
        place(source, args.output / safe_relative(relative), identity(source),
              hardlink=relative.startswith("artifacts/model/"))
    write_json(args.output / "artifacts/runtime/worker.json", worker_configuration())
    finish_stage(args.output)


def stage(args) -> None:
    claim_directory(args.output)
    record = read(args.execution / "execution.json")
    if record.get("schema_version") != 1 or record.get("status") != "unmeasured_provisional_execution":
        raise ValueError("Only the explicit provisional execution format is supported")
    files = record["pack_files"]
    names = [item["file"] for item in files]
    if len(names) != len(set(names)) or set(name for name in names if name.startswith("artifacts/model/")) != {
            "artifacts/model/" + name for name in MODEL_FILES}:
        raise ValueError("Execution closure has duplicate paths or missing model files")
    verify_model(args.model_dir)
    for item in files:
        relative = safe_relative(item["file"])
        if item["source"] != item["file"] or not (item["file"] == "descriptor.json" or item["file"].startswith("artifacts/")):
            raise ValueError("Execution source must be the same safe pack-relative artifact")
        source = args.model_dir / relative.name if item["file"].startswith("artifacts/model/") else args.execution / relative
        place(source, args.output / relative, item, hardlink=item["file"].startswith("artifacts/model/"))
    compact = record["descriptor_canonical"]
    if compact["file"] != CANONICAL:
        raise ValueError("Unexpected canonical descriptor path")
    place(args.execution / CANONICAL, args.output / CANONICAL, compact)
    finish_stage(args.output)
    if read(args.output / "execution.json") != record:
        raise ValueError("Reconstructed execution closure differs from the transported plan")


def accepted(args) -> None:
    validation = read(args.validation)
    benchmark = args.benchmark_dir / "benchmark.json"
    supervisor = args.benchmark_dir / "supervisor.json"
    if validation.get("schema_version") != 1 or validation.get("status") != "passed" or validation.get("performance") != PERFORMANCE:
        raise ValueError("Independent benchmark validator did not accept the agreed performance provenance")
    for field, path in (("benchmark_report", benchmark), ("supervisor_report", supervisor)):
        expected = validation.get(field, {})
        if expected.get("sha256") != sha(path) or expected.get("size_bytes") != path.stat().st_size:
            raise ValueError("Accepted evidence digest mismatch: " + field)
    if validation.get("workload_sha256") != sha(args.execution / PERFORMANCE["methodology"]):
        raise ValueError("Accepted workload identity differs")
    actual = [{"file": item["file"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]}
              for item in closure(args.execution)]
    if sorted(validation.get("execution_artifacts", []), key=lambda item: item["file"]) != sorted(actual, key=lambda item: item["file"]):
        raise ValueError("Accepted execution closure differs from the current exact files")
    original = read(args.preview_profile)
    if original.get("measurements") != {"performance": PERFORMANCE, "quality": QUALITY} or any(key in original for key in ("inputs", "signing", "license_acceptance")):
        raise ValueError("Preview profile contains unsupported changes")
    value = copy.deepcopy(original)
    value["measurements"] = {"performance": validation["performance"], "quality": QUALITY}
    value["inputs"] = {name: {"path": str(path.resolve(strict=True)), "digest": "sha256:" + sha(path), "license": "NOASSERTION"}
                       for name, path in (("native-benchmark", benchmark), ("benchmark-supervisor", supervisor),
                                          ("native-measurement-validation", args.validation))}
    write_json(args.output, value)


def verify_materialized(args) -> None:
    expected = read(args.execution / "execution.json")
    actual = closure(args.pack)
    expected_map = {item["file"]: item for item in expected["pack_files"]}
    actual_map = {item["file"]: item for item in actual}
    if set(actual_map) != set(expected_map) | set(REPORT_PATHS.values()):
        raise ValueError("Final materializer added, removed or renamed an unexpected execution artifact")
    for name, item in expected_map.items():
        verify(args.execution / safe_relative(name), item)
        verify(args.pack / safe_relative(name), item)
    accepted = read(args.accepted_profile)
    if set(accepted.get("inputs", {})) != set(REPORT_PATHS):
        raise ValueError("Accepted profile report inputs differ")
    for name, relative in REPORT_PATHS.items():
        supplied = accepted["inputs"][name]
        source = args.accepted_profile.parent / supplied["path"]
        expected_report = identity(source)
        if supplied["digest"] != "sha256:" + expected_report["sha256"]:
            raise ValueError("Accepted report source changed after validation")
        verify(args.pack / relative, expected_report)
    descriptor_pair(args.pack / "descriptor.json", args.execution / CANONICAL)
    manifest = read(args.pack / "manifest.json")
    if "signatures" in manifest or manifest.get("effective_license") != "NOASSERTION":
        raise ValueError("This evaluation recipe expects an explicitly unsigned NOASSERTION diagnostic")
    manifest_paths = set()
    for item in manifest["artifacts"]:
        name = item["path"]
        if name in manifest_paths or name not in actual_map:
            raise ValueError("Manifest has a duplicate or unexpected artifact")
        manifest_paths.add(name)
        if item["digest"] != "sha256:" + actual_map[name]["sha256"] or item["size"] != actual_map[name]["size_bytes"]:
            raise ValueError("Materialized manifest artifact hash mismatch")
    if manifest_paths != set(actual_map) or [item["path"] for item in manifest["artifacts"] if item["role"] == "runtime"] != [RUNTIME]:
        raise ValueError("Manifest artifact closure or native runtime role differs")
    if manifest.get("descriptor_digest") != "sha256:" + sha(args.execution / CANONICAL):
        raise ValueError("Canonical manifest descriptor digest mismatch")
    write_json(args.output, {"schema_version": 1, "status": "passed", "execution_artifacts_byte_identical": True,
                            "execution_artifact_count": len(expected_map), "manifest": identity(args.pack / "manifest.json"),
                            "signed": False, "full_conformance": False,
                            "remaining_gate": "Run the independent C ABI functional consumer on this final pack on the target Linux AArch64 host."})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profile")
    for name in ("blueprint", "model-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.set_defaults(action=profile)
    p = sub.add_parser("prepare")
    for name in ("model-dir", "worker-libs", "descriptor", "canonical-descriptor", "native-library",
                 "native-build-report", "compiler-report", "native-license", "workload", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.set_defaults(action=prepare)
    p = sub.add_parser("stage")
    for name in ("execution", "model-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.set_defaults(action=stage)
    p = sub.add_parser("accepted-profile")
    for name in ("preview-profile", "validation", "benchmark-dir", "execution", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.set_defaults(action=accepted)
    p = sub.add_parser("verify-materialized")
    for name in ("execution", "pack", "accepted-profile", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.set_defaults(action=verify_materialized)
    args = parser.parse_args()
    args.action(args)
    print(json.dumps({"status": "complete", "command": args.command, "output": str(args.output)}))


if __name__ == "__main__":
    main()
