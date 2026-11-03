#!/usr/bin/env python3
"""Compile an exported SKaiNET cached LFM2 graph for Vulkan, without CPU fallback.

This prepares a runtime bundle, not a parity or performance qualification.
Every attempt uses a new output directory and retains commands and compiler logs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BOUNDED_COMPILER_FLAGS = [
    "--iree-stream-resource-max-allocation-size=67108864",
    "--iree-stream-resource-max-range=67108864",
    "--iree-stream-resource-alias-mutable-bindings=false",
    "--iree-scheduling-optimize-bindings=false",
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_file(root, record):
    path = (root / record["file"]).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or digest(path) != record["sha256"]:
        raise ValueError("Export artifact identity or containment check failed")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler-wrapper", required=True, type=Path)
    parser.add_argument("--range-auditor", required=True, type=Path,
                        help="Source IREE-tools verify_vulkan_ranges.py; runs inside the pinned compiler image")
    parser.add_argument("--image", required=True, help="Existing pinned IREE compiler image ID")
    args = parser.parse_args()
    source = args.export.resolve(strict=True)
    wrapper = args.compiler_wrapper.resolve(strict=True)
    auditor = args.range_auditor.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Use an independent output directory")
    output.mkdir(parents=True, exist_ok=False)
    # Compile and audit immutable snapshots of the exact selected tools.
    wrapper_snapshot = output / "compiler-entrypoint.sh"
    auditor_snapshot = output / "verify_vulkan_ranges.py"
    driver_snapshot = output / "compile-driver.py"
    shutil.copyfile(wrapper, wrapper_snapshot)
    shutil.copyfile(auditor, auditor_snapshot)
    shutil.copyfile(Path(__file__), driver_snapshot)
    report = {"schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "started", "GPU_execution": False, "cpu_model_fallback": False,
              "commands": [], "compiler_wrapper_sha256": digest(wrapper_snapshot), "range_auditor_sha256": digest(auditor_snapshot),
              "compiler_driver_sha256": digest(driver_snapshot)}
    try:
        manifest_path = source / "export-manifest.json"
        export = json.loads(manifest_path.read_text())
        if export["function"] != "lfm2_cached_step" or export["status"] != "exported_not_yet_compiled":
            raise ValueError("Expected a complete cached-step export")
        mlir = checked_file(source, export["module"])
        parameters = checked_file(source, export["parameters"])
        contract_file = checked_file(source, export["provenance"]["generation_contract"])
        contract = json.loads(contract_file.read_text())
        if contract["rms_reduction_policy"] != "BALANCED_FP32" or contract["attention_layout_policy"] != "DIRECT_KV_PER_HEAD":
            raise ValueError("Unsupported numerical policy")
        if contract["embedding_strategy"] != "one_hot_matmul_contiguous_row_shards_tied_head":
            raise ValueError("Unsupported GPU embedding policy")
        if contract["embedding_shard_rows"] != [8192] * 8:
            raise ValueError("Expected eight 8192-row embedding shards for bounded initialization grids")
        for name, ports in (("input", contract["inputs"]), ("output", contract["outputs"])):
            if export[name + "_order"] != [p["name"] for p in ports] or export[name + "_shapes"] != [p["shape"] for p in ports]:
                raise ValueError("Export and generation contract disagree")
        inspected = subprocess.run(["docker", "image", "inspect", args.image], capture_output=True, text=True, check=True, timeout=30)
        image = json.loads(inspected.stdout)[0]
        if image["Config"]["Labels"].get("iree.version") != "3.11.0":
            raise ValueError("Expected the existing IREE 3.11.0 compiler image")
        report.update(image_id=image["Id"], export_manifest_sha256=digest(manifest_path),
                      generation_contract_sha256=digest(contract_file),
                      source_module_sha256=digest(mlir), source_parameters_sha256=digest(parameters))
        base = ["docker", "run", "--rm", "--network=none", "-v", f"{source}:/source:ro",
                "-v", f"{output}:/work", "-v", f"{wrapper_snapshot}:/entrypoint-new.sh:ro", "-w", "/work",
                "--entrypoint", "bash", image["Id"], "/entrypoint-new.sh"]
        operations = [
            ("parameters", ["convert-parameters", "--parameters=/source/" + parameters.relative_to(source).as_posix(), "--output=/work/model.irpa"]),
            ("vulkan", ["compile-vulkan", "/source/" + mlir.relative_to(source).as_posix(), "--target", "generic", "--out", "/work/model.vmfb",
                        "--iree-input-type=stablehlo", "--iree-input-demote-f32-to-f16=false", *BOUNDED_COMPILER_FLAGS]),
        ]
        for name, arguments in operations:
            command = base + arguments
            entry = {"name": name, "arguments": arguments, "log": name + ".log"}
            report["commands"].append(entry)
            with (output / entry["log"]).open("x") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            entry["exit_code"] = process.returncode
            if process.returncode:
                raise RuntimeError(f"{name} failed with exit code {process.returncode}; see {entry['log']}")
            print(json.dumps(entry), flush=True)
        binary = (output / "model.vmfb").read_bytes()
        if b"vulkan-spirv-fb" not in binary or b"embedded-elf" in binary:
            raise ValueError("Compiled bundle must contain Vulkan and no embedded CPU executable")
        audit_arguments = ["--vmfb", "/work/model.vmfb", "--output", "/work/vulkan-range-audit.json",
                           "--compiler-pin", image["Id"], "--max-binding-bytes", "67108864",
                           "--max-allocation-bytes", "8589934592", "--max-workgroup-counts", "65535", "65535", "65535"]
        audit_command = ["docker", "run", "--rm", "--network=none", "-v", f"{output}:/work",
                         "-v", f"{auditor_snapshot}:/verify_vulkan_ranges.py:ro", "--entrypoint", "python3",
                         image["Id"], "/verify_vulkan_ranges.py", *audit_arguments]
        audit_entry = {"name": "range-audit", "arguments": audit_arguments, "log": "range-audit.log"}
        report["commands"].append(audit_entry)
        with (output / audit_entry["log"]).open("x") as log:
            process = subprocess.run(audit_command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        audit_entry["exit_code"] = process.returncode
        if process.returncode:
            raise RuntimeError("Generated Vulkan range/grid audit failed; see range-audit.log")
        audit_path = output / "vulkan-range-audit.json"
        audit = json.loads(audit_path.read_text())
        if (audit["status"] != "passed" or audit["findings"] or audit["unresolved"] or
                audit["inputs"]["vmfb"]["sha256"] != digest(output / "model.vmfb") or
                audit["inputs"]["auditor"]["sha256"] != report["range_auditor_sha256"]):
            raise ValueError("Range/grid audit result does not bind the compiled module and auditor")
        range_record = {"file": audit_path.name, "sha256": digest(audit_path), "size_bytes": audit_path.stat().st_size}
        report["vulkan_range_audit"] = range_record
        print(json.dumps(audit_entry), flush=True)
        runtime = dict(contract)
        runtime.update(protocol="skainet.lfm2.iree.v1", backend="vulkan-spirv", dtype="float32",
                       embedding_policy="one_hot_sharded", parameter_scope="model")
        for kind, name in (("module", "model.vmfb"), ("parameters", "model.irpa")):
            path = output / name
            runtime[kind] = {"file": name, "sha256": digest(path), "size_bytes": path.stat().st_size}
        runtime["export_manifest_sha256"] = digest(manifest_path)
        runtime["source_generation_contract_sha256"] = digest(contract_file)
        runtime["vulkan_range_audit"] = range_record
        (output / "runtime-manifest.json").write_text(json.dumps(runtime, indent=2, allow_nan=False) + "\n")
        shutil.copyfile(manifest_path, output / "source-export-manifest.json")
        shutil.copyfile(contract_file, output / "generation-contract.json")
        report.update(status="compiled_not_executed", runtime_manifest_sha256=digest(output / "runtime-manifest.json"),
                      module=runtime["module"], parameters=runtime["parameters"])
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "compilation-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
