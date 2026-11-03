#!/usr/bin/env python3
"""Prepare an unqualified full-generation Vulkan execution candidate. Never authorizes materialization."""
from __future__ import annotations
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
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


PERFORMANCE = {"measured_on": "Linux AArch64 Vulkan, external Java 21 and Python 3.12",
               "methodology": "artifacts/evaluation/gpu-qualification.json"}
QUALITY = {"unmeasured": True, "harness": "Complete GPU target generation and task quality remain unqualified."}
RUNTIME = "artifacts/runtime/liblfm2_cartridge.so"
CANONICAL = ".provenance/descriptor.canonical.json"

def settings(path: Path) -> dict:
    value = read(path)
    if set(value) != {"model_directory", "wheel_directory", "device_uri"}:
        raise ValueError("Private settings require model_directory, wheel_directory and explicit device_uri only")
    if not isinstance(value["device_uri"], str) or not value["device_uri"].startswith("vulkan://") or not value["device_uri"][9:]:
        raise ValueError("Select an explicit Vulkan device URI; no CPU fallback exists")
    for name in ("model_directory", "wheel_directory"):
        value[name] = Path(value[name]).resolve(strict=True)
        if not value[name].is_dir(): raise ValueError("Expected local directory: " + name)
    return value

def profile(args):
    source = settings(args.settings)
    bp = read(args.blueprint)
    verify_model(source["model_directory"])
    wheel_sources = read(args.wheels)["wheels"]
    mirrors = {"liquid-lfm25-350m": source["model_directory"].as_uri() + "/"}
    for item in wheel_sources:
        verify(source["wheel_directory"] / item["file"], item)
        mirrors[item["source"]] = source["wheel_directory"].as_uri() + "/"
    value = {"spec_version": "0.4", "kind": "materialization-profile",
             "blueprint": {"id": bp["id"], "version": bp["version"], "digest": "sha256:" + sha(args.blueprint)},
             "target": "linux-aarch64-vulkan", "cartridge": {"id": bp["id"] + "-linux-aarch64-vulkan", "version": bp["version"]},
             "mirrors": mirrors, "measurements": {"performance": PERFORMANCE, "quality": QUALITY}}
    write_json(args.output, value)
    write_json(args.output.with_name("preview-status.json"), {"status":"unmeasured_descriptor_preview", "GPU_execution":False,
               "materialization_authorized":False, "device_uri":source["device_uri"]})

def descriptor_pair(raw: Path, compact: Path):
    value = read(raw)
    if canonical(value) != compact.read_bytes(): raise ValueError("Descriptor encodings disagree")
    if value.get("license") != "NOASSERTION" or value.get("performance") != PERFORMANCE: raise ValueError("Unexpected descriptor provenance")
    if value.get("target",{}).get("accelerator") != "vulkan" or value.get("attributes",{}).get("context_window_tokens") != 64:
        raise ValueError("Expected the explicit context64 Vulkan descriptor")
    return value

def extract_wheels(directory: Path, pins: Path, output: Path):
    entries = read(pins)
    if entries.get("python") != "3.12" or entries.get("platform") != "linux-aarch64": raise ValueError("Unexpected Python ABI")
    for record in entries["wheels"]:
        wheel = directory / safe_relative(record["file"]); verify(wheel, record)
        with zipfile.ZipFile(wheel) as archive:
            seen = set()
            for member in archive.infolist():
                name = member.filename.rstrip("/"); relative = safe_relative(name)
                if name in seen: raise ValueError("Duplicate wheel path")
                seen.add(name)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode): raise ValueError("Wheel symlinks are not supported")
                if member.file_size > 128 * 1024 * 1024: raise ValueError("Wheel member exceeds bounded extraction size")
                target = output / relative
                if member.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
                # Namespace directories may overlap, files may not. Never execute wheel installers.
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    write_json(output.parent / "python-wheels.json", entries)

def verify_bundle(directory: Path):
    runtime = read(directory / "runtime-manifest.json")
    if runtime.get("backend") != "vulkan-spirv" or runtime.get("dtype") != "float32" or runtime.get("capacity") != 64:
        raise ValueError("Expected complete cached FP32 Vulkan generation at capacity64")
    if runtime.get("embedding_shard_rows") != [8192] * 8: raise ValueError("Expected bounded eight-shard source policy")
    if runtime.get("function_name") != "lfm2_cached_step" or len(runtime.get("outputs", [])) != 23:
        raise ValueError("Complete decoder logits and all22 hybrid state ports are required")
    for key in ("module", "parameters"):
        item = runtime[key]; verify(directory / safe_relative(item["file"]), item)
    report = read(directory / "compilation-report.json")
    if report.get("status") != "compiled_not_executed" or report.get("cpu_model_fallback") is not False:
        raise ValueError("Expected actual no-fallback compilation provenance")
    if report["runtime_manifest_sha256"] != sha(directory / "runtime-manifest.json"): raise ValueError("Runtime manifest changed")
    if report["compiler_wrapper_sha256"] != sha(directory / "compiler-entrypoint.sh"): raise ValueError("Compiler wrapper changed")
    if report["compiler_driver_sha256"] != sha(directory / "compile-driver.py"): raise ValueError("Compiler driver changed")
    audit_record = runtime.get("vulkan_range_audit")
    if not audit_record or audit_record != report.get("vulkan_range_audit"): raise ValueError("Missing matching range-audit provenance")
    audit_path = directory / safe_relative(audit_record["file"]); verify(audit_path, audit_record)
    audit = read(audit_path)
    if audit.get("status") != "passed" or audit.get("findings") or audit.get("unresolved"):
        raise ValueError("Exact compiled Vulkan ranges/grids are unverified")
    if audit["inputs"]["vmfb"]["sha256"] != sha(directory / runtime["module"]["file"]): raise ValueError("Audit binds another VMFB")
    if audit["inputs"]["auditor"]["sha256"] != report["range_auditor_sha256"] or report["range_auditor_sha256"] != sha(directory / "verify_vulkan_ranges.py"):
        raise ValueError("Range auditor source changed")
    if audit.get("policy") != {"max_binding_bytes":67108864,"max_allocation_bytes":8589934592,"max_workgroup_counts":[65535,65535,65535]}:
        raise ValueError("Compiled range/grid policy differs")
    for dump in audit["inputs"]["dumps"]:
        artifact = dump["artifact"]
        verify(directory / Path(artifact["file"]).name, artifact)
    exported_path = directory / "source-export-manifest.json"
    contract_path = directory / "generation-contract.json"
    exported = read(exported_path); contract = read(contract_path)
    if report["export_manifest_sha256"] != sha(exported_path) or runtime["export_manifest_sha256"] != sha(exported_path):
        raise ValueError("Export provenance identity differs")
    if report["generation_contract_sha256"] != sha(contract_path) or runtime["source_generation_contract_sha256"] != sha(contract_path):
        raise ValueError("Generation contract identity differs")
    if any(runtime.get(key) != value for key,value in contract.items()): raise ValueError("Runtime and source tensor contracts differ")
    if exported["provenance"]["checkpoint_sha256"] != MODEL_FILES["model.safetensors"][1]: raise ValueError("Wrong source checkpoint")
    return runtime

def worker_configuration(device: str):
    return {"backend":"iree-vulkan", "metadata_dir":"artifacts/metadata", "bundle_dir":"artifacts/gpu",
            "python_package_dir":"artifacts/python", "python_executable_env":"LFM2_PYTHON_EXECUTABLE", "device_uri":device,
            "context_window_tokens":64,"checkpoint_sha256":MODEL_FILES["model.safetensors"][1],
            "metadata_sha256":{n:v[1] for n,v in MODEL_FILES.items() if n != "model.safetensors"},
            "decoder_policy":"HUGGING_FACE_5_COMPAT","default_max_output_tokens":8,"max_output_tokens":32}

def prepare(args):
    source = settings(args.settings); claim_directory(args.output)
    descriptor = descriptor_pair(args.descriptor,args.canonical_descriptor)
    validate_build(args.native_library,args.native_build_report,args.descriptor,args.canonical_descriptor)
    verify_model(source["model_directory"]); verify_bundle(args.bundle)
    mappings = {"descriptor.json":args.descriptor,CANONICAL:args.canonical_descriptor,RUNTIME:args.native_library,
                "artifacts/runtime/native-build.json":args.native_build_report,"artifacts/runtime/compiler-container.json":args.compiler_report,
                "artifacts/runtime/LICENSE.upstream":args.native_license}
    jars = sorted(args.worker_libs.iterdir())
    if not jars or any(p.suffix != ".jar" or not p.is_file() or p.is_symlink() for p in jars): raise ValueError("Expected only worker JAR build products")
    if not any("sk/ainet/cartridge/generator/lfm2/worker/MainKt.class" in zipfile.ZipFile(p).namelist() for p in jars):
        raise ValueError("Shared worker main class missing")
    mappings.update({"artifacts/jvm/lib/"+p.name:p for p in jars})
    mappings.update({"artifacts/metadata/"+n:source["model_directory"]/n for n in MODEL_FILES if n != "model.safetensors"})
    mappings.update({"artifacts/gpu/"+p.name:p for p in args.bundle.iterdir() if p.is_file()})
    for relative,path in mappings.items(): place(path,args.output/safe_relative(relative),identity(path))
    extract_wheels(source["wheel_directory"],args.wheels,args.output/"artifacts/python")
    write_json(args.output/"artifacts/runtime/worker.json",worker_configuration(source["device_uri"]))
    entries = closure(args.output)
    write_json(args.output/"execution.json",{"schema_version":1,"status":"unqualified_full_gpu_candidate","GPU_execution":False,
        "cpu_model_fallback":False,"pack_files":entries,"descriptor_canonical":{"file":CANONICAL,**identity(args.output/CANONICAL)},
        "external_dependencies":{"java":"JAVA_HOME/bin/java (21, jdk.incubator.vector)","python":"LFM2_PYTHON_EXECUTABLE (3.12)","vulkan_device":source["device_uri"]},
        "pending_gates":["full GPU numerical/state/generation parity","actual frontend and final C ABI generation","exact-artifact target measurement","independent acceptance","actual materialization and final-pack revalidation"]})
    write_json(args.output/"manifest.json",{"schema_version":"evaluation-execution-v1","kind":"unsigned-evaluation-stage",
        "id":descriptor["id"],"version":descriptor["version"],"effective_license":"NOASSERTION","digest_alg":"sha256",
        "descriptor_digest":"sha256:"+sha(args.output/CANONICAL),"signatures_present":False,"status":"unqualified_full_gpu_candidate",
        "artifacts":[{"path":x["file"],"digest":"sha256:"+x["sha256"],"size":x["size_bytes"],
                      "role":"runtime" if x["file"]==RUNTIME else "descriptor" if x["file"]=="descriptor.json" else
                             "weights" if x["file"]=="artifacts/gpu/model.irpa" else "model" if x["file"]=="artifacts/gpu/model.vmfb" else "other",
                      "license":MODEL_LICENSE if x["file"].startswith(("artifacts/metadata/","artifacts/gpu/")) else "NOASSERTION"} for x in entries]})

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("profile")
    for name in ("settings","blueprint","wheels","output"):p.add_argument("--"+name,type=Path,required=True)
    p.set_defaults(action=profile)
    p=sub.add_parser("prepare")
    for name in ("settings","wheels","worker-libs","bundle","descriptor","canonical-descriptor","native-library","native-build-report","compiler-report","native-license","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.set_defaults(action=prepare)
    args=parser.parse_args();args.action(args);print(json.dumps({"status":"complete","command":args.command,"qualified":False}))

if __name__ == "__main__": main()
