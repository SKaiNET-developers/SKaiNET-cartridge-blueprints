#!/usr/bin/env python3
"""Measure the pinned short-chat C ABI workload on Linux, with an owned watchdog.

Usage: benchmark_native_cartridge.py --pack PACK --library LIB --workload JSON
       --output NEW_DIRECTORY [--timeout-seconds 1800]

One loaded cartridge serves three warmups and thirty measured fresh sessions.
benchmark.json holds observations; supervisor.json attests exit/timeout handling.
Per-sample files preserve progress if a native call blocks. This is an application
latency/RSS benchmark, not a kernel throughput or model-quality evaluation.
"""
from __future__ import annotations

import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
from consume_native_cartridge import Buffer, Preflight, bind

WORKLOAD_SHA256 = "182bfd2b739ad977e732a196980cedf7247484bf1e1e8b64a272cd6bac41739c"
WORKER_MAIN = "sk.ainet.cartridge.generator.lfm2.worker.MainKt"
ORDER = ["arithmetic", "system", "multiturn"]


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(document):
    return json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def request_bytes(request):
    # Identical request serialization to the independent ABI consumer.
    return json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def save(path, document):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_workload(path):
    require(sha_file(path) == WORKLOAD_SHA256, "Workload identity differs from the pinned source")
    work = json.loads(Path(path).read_bytes())
    require(work["warmup_order"] == ORDER and work["measured_order"] == ORDER * 10, "Workload order differs")
    require([c["id"] for c in work["cases"]] == ORDER, "Workload case IDs differ")
    for case in work["cases"]:
        require(case["request"]["max_output_tokens"] == 8, "Output budget differs")
        require(hashlib.sha256(request_bytes(case["request"])).hexdigest() == case["request_sha256"], "Request bytes differ")
    return work


def file_record(path, relative):
    path = Path(path)
    record = {"file": str(relative), "sha256": sha_file(path), "size_bytes": path.stat().st_size}
    if path.is_symlink():
        record.update(symlink=os.readlink(path), resolved=str(path.resolve(strict=True)))
    return record


def tree_inventory(root):
    root = Path(root)
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and path.is_dir():
            raise ValueError(f"Directory symlink is not an audited file: {path.relative_to(root)}")
        if path.is_file():
            result.append(file_record(path, path.relative_to(root)))
        elif not path.is_dir():
            raise ValueError(f"Unsupported or missing artifact: {path.relative_to(root)}")
    return result


def java_home_for(pack):
    packaged = pack / "artifacts/runtime/java"
    if packaged.exists() or packaged.is_symlink():
        require(packaged.is_file() and os.access(packaged, os.X_OK), "Packaged Java exists but is unusable; refusing fallback")
        candidate = packaged
    else:
        require(bool(os.environ.get("JAVA_HOME")), "JAVA_HOME is required when packaged Java is absent")
        candidate = Path(os.environ["JAVA_HOME"]) / "bin/java"
    require(candidate.is_file() and os.access(candidate, os.X_OK), "No packaged or JAVA_HOME Java executable")
    java = candidate.resolve(strict=True)
    home = java.parent.parent
    require((home / "release").is_file() and (home / "lib/modules").is_file() and
            (home / "lib/server/libjvm.so").is_file(), "Expected Linux Java runtime layout is missing")
    return java, home


def audit_artifacts(pack, library, workload):
    """Hash full pack and full external JRE; all work is outside sample timing."""
    pack, library = Path(pack).resolve(strict=True), Path(library).resolve(strict=True)
    files = tree_inventory(pack)
    by_path = {f["file"]: f for f in files}
    manifest = json.loads((pack / "manifest.json").read_bytes())
    require(manifest.get("digest_alg") == "sha256", "Manifest digest algorithm differs")
    artifacts = manifest["artifacts"]
    require(len(artifacts) == len({a["path"] for a in artifacts}), "Duplicate manifest path")
    for artifact in artifacts:
        actual = by_path.get(artifact["path"])
        require(actual is not None and artifact["digest"] == "sha256:" + actual["sha256"] and
                artifact["size"] == actual["size_bytes"], f"Manifest artifact differs: {artifact['path']}")
    descriptor_digest = hashlib.sha256(canonical(json.loads((pack / "descriptor.json").read_bytes()))).hexdigest()
    require(manifest["descriptor_digest"] == "sha256:" + descriptor_digest, "Canonical descriptor digest differs")
    runtime = [a for a in artifacts if a["role"] == "runtime"]
    require(len(runtime) == 1 and (pack / runtime[0]["path"]).resolve(strict=True) == library, "Library is not the manifest runtime")
    config = json.loads((pack / "artifacts/runtime/worker.json").read_bytes())
    require(config["decoder_policy"] == workload["decoder_policy"], "Decoder policy differs")
    require(config["checkpoint_sha256"] == workload["model"]["checkpoint_sha256"] and
            config["metadata_sha256"] == workload["model"]["metadata_sha256"], "Worker model pins differ")
    require(config["max_output_tokens"] >= 8, "Package output limit is too small")
    model = pack / config["model_dir"]
    require(model.resolve(strict=True).is_relative_to(pack), "Model directory escapes package")
    for name, expected in dict(config["metadata_sha256"], **{"model.safetensors": config["checkpoint_sha256"]}).items():
        record = by_path.get(str((model / name).relative_to(pack)))
        require(record is not None and record["sha256"] == expected, f"Model bytes differ: {name}")
    java, home = java_home_for(pack)
    return {"pack_files": files, "manifest_sha256": by_path["manifest.json"]["sha256"],
            "descriptor_raw_sha256": by_path["descriptor.json"]["sha256"], "descriptor_canonical_sha256": descriptor_digest,
            "java_executable": str(java), "java_home": str(home),
            "java_files": tree_inventory(home), "library_relative_path": runtime[0]["path"]}


def read_text(path):
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return None


def board_state():
    """Public machine state only: deliberately excludes hostname/IP/serial IDs."""
    cpu_models = sorted({line.split(":", 1)[1].strip() for line in (read_text("/proc/cpuinfo") or "").splitlines()
                         if line.startswith(("model name", "Hardware", "CPU part")) and ":" in line})
    thermal = [{"zone": p.parent.name, "type": read_text(p.parent / "type"), "millidegrees_c": read_text(p)}
               for p in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))]
    frequency = [{"policy": p.name, **{name: read_text(p / name) for name in
                  ("affected_cpus", "scaling_governor", "scaling_cur_freq", "scaling_min_freq", "scaling_max_freq")}}
                 for p in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*"))]
    power = [{"supply": p.name, **{name: read_text(p / name) for name in
              ("type", "status", "online", "voltage_now", "current_now", "power_now")}}
             for p in sorted(Path("/sys/class/power_supply").glob("*"))]
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "system": platform.system(),
            "machine": platform.machine(), "kernel": platform.release(), "os_release": read_text("/etc/os-release"),
            "board_model": (read_text("/proc/device-tree/model") or "").rstrip("\0"),
            "logical_cpus": os.cpu_count(), "cpu_models": cpu_models,
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "loadavg": read_text("/proc/loadavg"), "meminfo": read_text("/proc/meminfo"),
            "uptime": read_text("/proc/uptime"), "thermal": thermal, "cpufreq": frequency, "power": power}


def proc_record(pid, proc=Path("/proc")):
    directory = proc / str(pid)
    # comm may itself contain spaces or parentheses; stat fields start after its last ')'.
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    status = {}
    for line in (directory / "status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, unit = line.split()
            require(unit == "kB", "Unexpected proc memory unit")
            status[key[:-1] + "_bytes"] = int(value) * 1024
    require("VmRSS_bytes" in status and "VmHWM_bytes" in status, "Process has exited or lacks resident memory accounting")
    return {"pid": pid, "ppid": int(fields[1]), "process_group": int(fields[2]),
            "session_id": int(fields[3]), "starttime_ticks": int(fields[19]),
            "cmdline": [p.decode("utf-8", "replace") for p in (directory / "cmdline").read_bytes().split(b"\0") if p],
            **status}


def is_worker(record, parent_pid, pack):
    args = record["cmdline"]
    return (record["ppid"] == parent_pid and WORKER_MAIN in args and "--pack-dir" in args and
            args.index("--pack-dir") + 1 < len(args) and args[args.index("--pack-dir") + 1] == str(pack))


class RssSampler:
    def __init__(self, pack, interval=0.1, proc=Path("/proc")):
        self.pack, self.interval, self.proc = Path(pack), interval, Path(proc)
        self.pid = os.getpid()
        self.phase = "startup"
        self.rows, self.errors, self.workers = [], [], {}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="benchmark-proc-rss", daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                consumer = proc_record(self.pid, self.proc)
                children_path = self.proc / str(self.pid) / "task" / str(self.pid) / "children"
                child_ids = [int(p) for p in children_path.read_text().split()]
                workers = []
                for pid in child_ids:
                    try:
                        record = proc_record(pid, self.proc)
                        if is_worker(record, self.pid, self.pack):
                            key = f"{pid}:{record['starttime_ticks']}"
                            self.workers[key] = record
                            workers.append(record)
                    except (OSError, ValueError, IndexError):
                        # A short-lived Java preflight can exit between proc reads.
                        continue
                self.rows.append({"monotonic_ns": time.monotonic_ns(), "phase": self.phase,
                                  "consumer": consumer, "workers": workers})
            except (OSError, ValueError, IndexError) as error:
                self.errors.append(repr(error))
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        require(not self.thread.is_alive(), "RSS monitor failed to stop")
        values = lambda process, key: [p[key] for r in self.rows for p in
                                      ([r["consumer"]] if process == "consumer" else r["workers"]) if key in p]
        return {"interval_ms": self.interval * 1000, "samples": self.rows, "errors": self.errors,
                "worker_identities": list(self.workers.values()),
                "peaks": {process: {key: max(values(process, key), default=None)
                                     for key in ("VmRSS_bytes", "VmHWM_bytes")}
                          for process in ("consumer", "jvm")}}


def error_detail(lib, handle):
    raw = lib.ctg_last_error(handle)
    return raw.decode("utf-8", "replace") if raw else None


def verify_events(events, expected):
    require(len(events) >= 3, "Missing Ready/Delta/Final events")
    require(events[0]["kind"] == 1 and events[0]["payload"] == {"type": "ready", "prompt_ids": expected["prompt_ids"]}, "Ready prompt IDs differ")
    final = events[-1]
    require(final["kind"] == 2 and final["payload"] == {"type": "final", "text": expected["text"],
            "finish_reason": expected["finish_reason"], "tokens": expected["tokens"]}, "Final text, tokens or reason differ")
    deltas = []
    for event in events[1:-1]:
        payload = event["payload"]
        require(event["kind"] == 1 and set(payload) == {"type", "text"} and payload["type"] == "delta" and
                isinstance(payload["text"], str) and payload["text"], "Malformed Delta event")
        deltas.append(payload["text"])
    require("".join(deltas) == expected["text"], "Delta text differs from final/reference")


def run_sample(lib, cartridge, case, sequence, phase, clock=time.monotonic_ns):
    raw = request_bytes(case["request"])
    memory = C.create_string_buffer(raw)
    view = Buffer(C.sizeof(Buffer), C.cast(memory, C.c_void_p), len(raw))
    session = C.c_void_p()
    record = {"sequence": sequence, "phase": phase, "case_id": case["id"], "request_sha256": case["request_sha256"],
              "status": "running", "events": [], "abi_calls": [], "boundaries_ns": {}}
    boundaries = record["boundaries_ns"]

    def checked(name, status, handle, timestamp):
        call = {"call": name, "status": status, "returned_monotonic_ns": timestamp}
        if status:
            call["detail"] = error_detail(lib, handle)
        record["abi_calls"].append(call)
        require(status == 0, f"{name} returned {status}: {call.get('detail')}")

    try:
        boundaries["start"] = clock()
        status = lib.ctg_session_begin(cartridge, None, C.byref(session))
        checked("session_begin", status, cartridge, clock())
        require(session.value, "session_begin returned no handle")
        checked("push", lib.ctg_push(session, C.byref(view)), session, clock())
        checked("flush", lib.ctg_flush(session), session, clock())
        for _ in range(1024):
            result = C.c_void_p()
            status = lib.ctg_pull(session, C.byref(result))
            returned = clock()
            try:
                checked("pull", status, session, returned)
                if not result.value:
                    boundaries["drain"] = returned
                    break
                payload = Buffer(C.sizeof(Buffer), None, 0)
                checked("result_get_payload", lib.ctg_result_get_payload(result, C.byref(payload)), None, clock())
                event = {"kind": lib.ctg_result_get_kind(result), "payload": json.loads(C.string_at(payload.data, payload.len)),
                         "returned_monotonic_ns": returned}
                record["events"].append(event)
                kind = event["payload"].get("type")
                if kind == "ready":
                    boundaries.setdefault("ready", returned)
                elif kind == "delta" and event["payload"].get("text"):
                    boundaries.setdefault("first_text", returned)
                elif kind == "final":
                    boundaries.setdefault("final", returned)
            finally:
                if result.value:
                    lib.ctg_result_release(result)
        else:
            raise ValueError("Stream failed to drain within 1024 events")
    except Exception as error:
        record["error"] = repr(error)
    finally:
        if session.value:
            status = lib.ctg_session_end(session)
            boundaries["end"] = clock()
            # session_end consumes a valid handle even on failure: never query it.
            try:
                checked("session_end", status, None, boundaries["end"])
            except Exception as error:
                record["cleanup_error"] = repr(error)
        else:
            boundaries["end"] = clock()
    if "error" not in record and "cleanup_error" not in record:
        try:
            verify_events(record["events"], case["expected"])
            require(all(k in boundaries for k in ("ready", "first_text", "final", "drain", "end")), "Missing timing boundary")
            require(boundaries["start"] <= boundaries["ready"] <= boundaries["first_text"] <= boundaries["final"] <= boundaries["drain"] <= boundaries["end"], "Timing order differs")
            final = record["events"][-1]["payload"]
            record.update(status="passed", exact_reference_trace=True, prompt_tokens=len(case["expected"]["prompt_ids"]),
                          generated_tokens=len(final["tokens"]), generated_token_count_includes_eos=True,
                          delta_events=len(record["events"]) - 2,
                          latency_ms={key: (boundaries[value] - boundaries["start"]) / 1e6 for key, value in
                                      (("e2e", "end"), ("ready", "ready"), ("ttft", "first_text"), ("final", "final"), ("drain", "drain"))},
                          final_to_drain_ms=(boundaries["drain"] - boundaries["final"]) / 1e6,
                          session_end_ms=(boundaries["end"] - boundaries["drain"]) / 1e6)
        except Exception as error:
            record["error"] = repr(error)
    if record["status"] != "passed":
        record["status"] = "failed"
    return record


def nearest_rank(values, percentile):
    require(bool(values) and 0 < percentile <= 1, "Invalid percentile input")
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def aggregate(samples):
    require(len(samples) == 30 and [s["case_id"] for s in samples] == ORDER * 10 and
            all(s["status"] == "passed" for s in samples), "Cannot publish complete percentiles for missing/failed samples")

    def stats(rows):
        return {"n": len(rows), "prompt_tokens": sum(s["prompt_tokens"] for s in rows),
                "generated_tokens_including_eos": sum(s["generated_tokens"] for s in rows),
                "latency_ms": {metric: {"p50": nearest_rank([s["latency_ms"][metric] for s in rows], .50),
                                        "p95": nearest_rank([s["latency_ms"][metric] for s in rows], .95)}
                               for metric in ("e2e", "ready", "ttft", "final", "drain")}}
    return {"percentile_method": "nearest_rank_ceil_p_times_n", "mix": "equal case weight, deterministic round robin",
            "all": stats(samples), "by_case": {label: stats([s for s in samples if s["case_id"] == label]) for label in ORDER}}


def execute(args):
    report = {"schema_version": 1, "scope": "native_abi_short_chat_performance", "status": "running",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "warmups": [], "samples": [],
              "performance_benchmark": True, "task_quality_qualified": False, "sealed_cartridge": False,
              "full_spec_conformance": False, "errors": []}
    cartridge, lib, sampler = C.c_void_p(), None, None
    work = None
    try:
        require(platform.system() == "Linux", "Benchmark RSS accounting requires Linux /proc")
        work = load_workload(args.workload)
        report.update(workload_sha256=sha_file(args.workload), workload=work,
                      performance={key: work[key] for key in ("measured_on", "methodology")},
                      sources=[file_record(Path(__file__), "benchmark_native_cartridge.py"),
                               file_record(Path(__file__).with_name("consume_native_cartridge.py"), "consume_native_cartridge.py")])
        report["artifacts_before"] = audit_artifacts(args.pack, args.library, work)
        report["board_before"] = board_state()
        lib = bind(args.library)
        require(lib.ctg_abi_version() == 1, "Unexpected ABI version")
        require(lib.ctg_descriptor_json() == canonical(json.loads((args.pack / "descriptor.json").read_bytes())), "Embedded canonical descriptor differs")
        sampler = RssSampler(args.pack)
        sampler.start()
        pf = Preflight(C.sizeof(Preflight), 0, None)
        started = time.monotonic_ns()
        status = lib.ctg_preflight(os.fsencode(args.pack), C.byref(pf))
        report["preflight"] = {"status": status, "ok": pf.ok, "duration_ms": (time.monotonic_ns() - started) / 1e6,
                               "detail": pf.detail.decode("utf-8", "replace") if pf.detail else None}
        require(status == 0 and pf.ok == 1, "Preflight failed")
        sampler.phase = "open_load"
        started = time.monotonic_ns()
        status = lib.ctg_open(os.fsencode(args.pack), None, C.byref(cartridge))
        report["open"] = {"status": status, "duration_ms": (time.monotonic_ns() - started) / 1e6,
                          "cache_state": "Prior full artifact hashing; no page-cache flush; not a cold-load measurement"}
        require(status == 0 and cartridge.value, f"Open failed: {status}, {error_detail(lib, None)}")
        by_id = {c["id"]: c for c in work["cases"]}
        for sequence, label in enumerate(work["warmup_order"] + work["measured_order"]):
            phase = "warmup" if sequence < 3 else "measured"
            sampler.phase = f"{phase}-{sequence:03d}-{label}"
            stem = f"sample-{sequence:03d}-{phase}-{label}"
            save(args.output / (stem + ".started.json"), {"sequence": sequence, "phase": phase,
                 "case_id": label, "monotonic_ns": time.monotonic_ns()})
            sample = run_sample(lib, cartridge, by_id[label], sequence, phase)
            save(args.output / (stem + ".json"), sample)
            report["warmups" if phase == "warmup" else "samples"].append(sample)
            require(sample["status"] == "passed", f"Sample {sequence} ({label}) failed")
        report["aggregates"] = aggregate(report["samples"])
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append(repr(error))
    finally:
        if lib is not None and cartridge.value:
            if sampler:
                sampler.phase = "close"
            started = time.monotonic_ns()
            status = lib.ctg_close(cartridge)
            report["close"] = {"status": status, "duration_ms": (time.monotonic_ns() - started) / 1e6}
            if status:
                # A failed close may retain its handle. Do not infer freed/live
                # state from a pointer; outer supervisor owns remaining processes.
                report["status"] = "failed"
                report["errors"].append(f"Close failed: {status}; owned supervisor must clean up")
        if sampler:
            try:
                report["rss"] = sampler.stop()
                require(len(report["rss"]["worker_identities"]) == 1, "Did not identify exactly one persistent worker PID/starttime")
                require(not report["rss"]["errors"], "RSS monitoring failed")
            except Exception as error:
                report["status"] = "failed"
                report["errors"].append(repr(error))
        try:
            report["board_after"] = board_state()
            if work is not None:
                report["artifacts_after"] = audit_artifacts(args.pack, args.library, work)
                require(report.get("artifacts_before") == report["artifacts_after"], "Artifacts changed during execution")
                require(sha_file(args.workload) == report["workload_sha256"], "Workload changed during execution")
                require(report["sources"] == [file_record(Path(__file__), "benchmark_native_cartridge.py"),
                        file_record(Path(__file__).with_name("consume_native_cartridge.py"), "consume_native_cartridge.py")], "Harness sources changed during execution")
                report["identities_unchanged"] = True
        except Exception as error:
            report["status"] = "failed"
            report["errors"].append(repr(error))
        # A late failure invalidates qualification; keep raw numbers as evidence,
        # but never offer them as a successful aggregate benchmark.
        if report["status"] != "passed":
            report.pop("aggregates", None)
        save(args.output / "benchmark.json", report)
    return 0 if report["status"] == "passed" else 1


def signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def supervise(command, output, timeout):
    start = time.monotonic()
    report = {"schema_version": 1, "scope": "native_abi_benchmark_owned_process", "status": "failed",
              "timeout_seconds": timeout, "timed_out": False, "command": command}
    with (output / "process.log").open("xb") as stream:
        try:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        except OSError as error:
            report.update(error=repr(error), exit_code=None, elapsed_seconds=time.monotonic() - start,
                          log=file_record(output / "process.log", "process.log"))
            save(output / "supervisor.json", report)
            return report
        report["owned_process_group"] = process.pid
        try:
            report["exit_code"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            report["timed_out"] = True
            signal_group(process.pid, signal.SIGTERM)
            try:
                report["exit_code"] = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_group(process.pid, signal.SIGKILL)
                try:
                    report["exit_code"] = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    report["exit_code"] = None
                    report["cleanup_error"] = "Owned process did not exit after SIGKILL deadline"
        try:
            os.killpg(process.pid, 0)
            report["remaining_process_group_after_controller_exit"] = True
        except ProcessLookupError:
            report["remaining_process_group_after_controller_exit"] = False
        # Always clean the owned process group after the controller exits, including
        # a failed FFI cleanup path. Never identify workers by name or killall.
        signal_group(process.pid, signal.SIGKILL)
    report["elapsed_seconds"] = time.monotonic() - start
    child_report = output / "benchmark.json"
    if child_report.is_file():
        report["benchmark_report"] = file_record(child_report, child_report.name)
        if (report.get("exit_code") == 0 and not report["timed_out"] and
                not report["remaining_process_group_after_controller_exit"] and
                json.loads(child_report.read_bytes()).get("status") == "passed"):
            report["status"] = "passed"
    report["log"] = file_record(output / "process.log", "process.log")
    save(output / "supervisor.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("Timeout must be finite and positive")
    for name in ("pack", "library", "workload"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.absolute()
    if args._child:
        return execute(args)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--_child", "--pack", str(args.pack),
               "--library", str(args.library), "--workload", str(args.workload), "--output", str(args.output)]
    result = supervise(command, args.output, args.timeout_seconds)
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
