#!/usr/bin/env python3
"""Independently recompute saved native benchmark results; never run inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from validate_native_cartridge import references

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["arithmetic", "system", "multiturn"]
PERFORMANCE = {
    "measured_on": "Linux AArch64 CPU, external Java 21",
    "methodology": "artifacts/evaluation/native-abi-workload.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    return {"sha256": sha_file(path), "size_bytes": Path(path).stat().st_size}


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def indexed(records):
    result = {}
    for record in records:
        name = record["file"]
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and name not in result,
                "Unsafe or duplicate artifact path")
        require(len(record["sha256"]) == 64 and type(record["size_bytes"]) is int and record["size_bytes"] >= 0,
                "Invalid artifact identity")
        result[name] = record
    return result


def check_workload(work):
    original = read(ROOT / "samples/native-abi-workload.json")
    require(work == original, "Workload differs from saved source")
    require(work["warmup_order"] == ORDER and work["measured_order"] == ORDER * 10,
            "Workload schedule differs")
    refs = references()
    require([c["id"] for c in work["cases"]] == ORDER, "Case order differs")
    for case in work["cases"]:
        ref = refs[case["id"]]
        require(case["expected"] == {
            "prompt_ids": ref["prompt_ids"], "tokens": ref["expected_generated_ids"],
            "text": ref["expected_decoded_completion_skip_special_tokens"], "finish_reason": "STOP"},
            "Workload reference differs")
        request_source = ROOT / case["original_request"]["file"]
        require(identity(request_source)["sha256"] == case["original_request"]["sha256"], "Original request changed")
        request = read(request_source)
        require(case["request"] == {"messages": request["messages"], "max_output_tokens": request["max_new_tokens"]},
                "Workload request changed")
        request_bytes = json.dumps(case["request"], ensure_ascii=False, separators=(",", ":")).encode()
        require(hashlib.sha256(request_bytes).hexdigest() == case["request_sha256"], "Request digest differs")
        require(sha_file(ROOT / case["reference"]["file"]) == case["reference"]["sha256"], "Reference source changed")
    return {case["id"]: case for case in work["cases"]}


def sample_metrics(sample, case, sequence, directory):
    label = case["id"]
    phase = "warmup" if sequence < 3 else "measured"
    require(sample["sequence"] == sequence and sample["phase"] == phase and sample["case_id"] == label,
            "Sample order differs")
    require(sample["status"] == "passed" and sample.get("exact_reference_trace") is True and
            not sample.get("error") and not sample.get("cleanup_error"), "Failed sample")
    require(sample["request_sha256"] == case["request_sha256"], "Sample request differs")
    stem = f"sample-{sequence:03d}-{phase}-{label}"
    require(read(directory / (stem + ".json")) == sample, "Per-sample evidence differs")
    marker = read(directory / (stem + ".started.json"))
    b = sample["boundaries_ns"]
    require(marker["sequence"] == sequence and marker["phase"] == phase and marker["case_id"] == label and
            marker["monotonic_ns"] <= b["start"], "Start marker differs")
    timestamps = [b[key] for key in ("start", "ready", "first_text", "final", "drain", "end")]
    require(all(type(t) is int and t >= 0 for t in timestamps) and timestamps == sorted(timestamps), "Timing order differs")
    events, expected = sample["events"], case["expected"]
    require(len(events) >= 3 and [e["kind"] for e in events] == [1] * (len(events) - 1) + [2], "Result kind order differs")
    require(events[0]["payload"] == {"type": "ready", "prompt_ids": expected["prompt_ids"]}, "Prompt IDs differ")
    require(events[-1]["payload"] == {"type": "final", "text": expected["text"], "tokens": expected["tokens"],
            "finish_reason": expected["finish_reason"]}, "Generated output differs")
    deltas = [e["payload"] for e in events[1:-1]]
    require(all(set(d) == {"type", "text"} and d["type"] == "delta" and isinstance(d["text"], str) and d["text"]
                for d in deltas), "Malformed Delta")
    require("".join(d["text"] for d in deltas) == expected["text"], "Streaming text differs")
    require([events[i]["returned_monotonic_ns"] for i in (0, 1, -1)] == [b["ready"], b["first_text"], b["final"]],
            "Timing boundaries differ from observed events")
    calls = sample["abi_calls"]
    expected_calls = ["session_begin", "push", "flush"] + [v for _ in events for v in ("pull", "result_get_payload")] + ["pull", "session_end"]
    require([c["call"] for c in calls] == expected_calls and all(c["status"] == 0 for c in calls), "ABI call order/status differs")
    call_times = [c["returned_monotonic_ns"] for c in calls]
    require(call_times == sorted(call_times) and b["start"] <= call_times[0] and call_times[-1] == b["end"], "ABI timestamps differ")
    pulls = [c["returned_monotonic_ns"] for c in calls if c["call"] == "pull"]
    require(pulls == [e["returned_monotonic_ns"] for e in events] + [b["drain"]], "Pull/event timing differs")
    latencies = {metric: (b[boundary] - b["start"]) / 1e6 for metric, boundary in
                 (("e2e", "end"), ("ready", "ready"), ("ttft", "first_text"), ("final", "final"), ("drain", "drain"))}
    require(sample["latency_ms"] == latencies and sample["final_to_drain_ms"] == (b["drain"] - b["final"]) / 1e6 and
            sample["session_end_ms"] == (b["end"] - b["drain"]) / 1e6, "Derived latency differs")
    require(sample["prompt_tokens"] == len(expected["prompt_ids"]) and sample["generated_tokens"] == len(expected["tokens"]) and
            sample["generated_token_count_includes_eos"] is True and sample["delta_events"] == len(deltas), "Token counts differ")
    return latencies


def stats(rows):
    result = {"n": len(rows), "prompt_tokens": sum(s["prompt_tokens"] for s in rows),
              "generated_tokens_including_eos": sum(s["generated_tokens"] for s in rows), "latency_ms": {}}
    for metric in ("e2e", "ready", "ttft", "final", "drain"):
        values = sorted(s["latency_ms"][metric] for s in rows)
        result["latency_ms"][metric] = {"p50": values[math.ceil(.5 * len(values)) - 1],
                                       "p95": values[math.ceil(.95 * len(values)) - 1]}
    return result


def check_rss(report, pack_argument, supervisor):
    rss = report["rss"]
    require(rss["interval_ms"] == 100 and not rss["errors"] and len(rss["worker_identities"]) == 1, "Invalid process accounting")
    worker = rss["worker_identities"][0]
    key = (worker["pid"], worker["starttime_ticks"])
    values = {process: {metric: [] for metric in ("VmRSS_bytes", "VmHWM_bytes")} for process in ("consumer", "jvm")}
    totals, times = [], []
    consumer_key = None
    for row in rss["samples"]:
        times.append(row["monotonic_ns"])
        consumer = row["consumer"]
        require(consumer["pid"] == consumer["process_group"] == consumer["session_id"] == supervisor["owned_process_group"] and
                consumer["cmdline"] == supervisor["command"], "Memory accounting is not the supervised consumer")
        current_key = (consumer["pid"], consumer["starttime_ticks"])
        consumer_key = consumer_key or current_key
        require(current_key == consumer_key and len(row["workers"]) <= 1, "Process identity changed")
        for proc in row["workers"]:
            args = proc["cmdline"]
            require((proc["pid"], proc["starttime_ticks"]) == key and proc["ppid"] == consumer["pid"] and
                    "sk.ainet.cartridge.generator.lfm2.worker.MainKt" in args and "--pack-dir" in args and
                    args[args.index("--pack-dir") + 1] == pack_argument, "Worker identity differs")
        for label, processes in (("consumer", [consumer]), ("jvm", row["workers"])):
            for proc in processes:
                for metric in values[label]:
                    require(type(proc[metric]) is int and proc[metric] > 0, "Invalid memory observation")
                    values[label][metric].append(proc[metric])
        totals.append(consumer["VmRSS_bytes"] + sum(p["VmRSS_bytes"] for p in row["workers"]))
    require(times and times == sorted(times), "RSS timestamps absent or unordered")
    worker_times = [row["monotonic_ns"] for row in rss["samples"] if row["workers"]]
    require(worker_times and min(worker_times) <= report["samples"][0]["boundaries_ns"]["start"] and
            max(worker_times) >= report["samples"][-1]["boundaries_ns"]["start"], "Worker observations do not cover measured run")
    peaks = {label: {metric: max(observations) for metric, observations in columns.items()} for label, columns in values.items()}
    require(rss["peaks"] == peaks, "Reported RSS peaks differ")
    return {"peaks": peaks, "observed_aggregate_rss_peak_bytes": max(totals), "sample_count": len(times),
            "limitation": "100ms /proc sampling; shared pages may be counted twice; transient peak may be missed."}


def check(directory, execution):
    report, supervisor = read(directory / "benchmark.json"), read(directory / "supervisor.json")
    require(report["schema_version"] == 1 and report["scope"] == "native_abi_short_chat_performance" and
            report["status"] == "passed" and not report["errors"], "Benchmark failed")
    require(supervisor["scope"] == "native_abi_benchmark_owned_process" and supervisor["status"] == "passed" and
            supervisor["exit_code"] == 0 and supervisor["timed_out"] is False and not supervisor.get("cleanup_error") and
            supervisor.get("remaining_process_group_after_controller_exit") is False, "Supervisor failed")
    for name, record in (("benchmark.json", supervisor["benchmark_report"]), ("process.log", supervisor["log"])):
        require(all(record[k] == v for k, v in identity(directory / name).items()), "Supervisor evidence digest differs")
    require(report["performance_benchmark"] is True and all(report[k] is False for k in
            ("task_quality_qualified", "sealed_cartridge", "full_spec_conformance")), "Overstated qualification")
    cases = check_workload(report["workload"])
    require(report["workload_sha256"] == sha_file(ROOT / "samples/native-abi-workload.json"), "Workload digest differs")
    require(report["performance"] == PERFORMANCE, "Performance provenance differs")
    for source in report["sources"]:
        require(all(source[k] == v for k, v in identity(ROOT / "scripts" / source["file"]).items()), "Harness source changed")
    require(len(report["sources"]) == 2 and {s["file"] for s in report["sources"]} ==
            {"benchmark_native_cartridge.py", "consume_native_cartridge.py"}, "Harness coverage differs")
    require(report["artifacts_before"] == report["artifacts_after"] and report["identities_unchanged"] is True, "Execution files changed")
    observed = indexed(report["artifacts_before"]["pack_files"])
    # The blueprint writes this inventory from actual build inputs. Recompute
    # every source identity here; hashes written into a plan are not sufficient.
    expected = read(execution / "execution.json")
    closure = []
    for item in expected["pack_files"]:
        indexed([item])
        require(item["source"] == item["file"], "Execution source must be the identical safe relative path")
        source = Path(item["source"])
        if not source.is_absolute():
            source = execution / source
        actual = identity(source)
        require(all(item[k] == v for k, v in actual.items()), "Expected execution source changed: " + item["file"])
        require(item["file"] in observed and all(observed[item["file"]][k] == v for k, v in actual.items()),
                "Measured execution differs: " + item["file"])
        closure.append({"file": item["file"], **actual})
    indexed(closure)
    expected_files = {str(p.relative_to(execution)): p for p in execution.rglob("*") if p.is_file()}
    require(set(observed) == set(expected_files), "Measured pack has missing or extra files")
    closure_by_path = {item["file"]: item for item in closure}
    require({name for name in observed if name == "descriptor.json" or name.startswith("artifacts/")} == set(closure_by_path),
            "Execution closure omits executable artifacts")
    for name, path in expected_files.items():
        require(not path.is_symlink() and "symlink" not in observed[name], "Execution contains unapproved symlink")
        actual = closure_by_path.get(name) or identity(path)
        require(all(observed[name][key] == actual[key] for key in ("sha256", "size_bytes")), "Staging identity differs: " + name)
    require(any(x["file"] == "descriptor.json" for x in closure) and any(x["file"].endswith("model.safetensors") for x in closure),
            "Expected closure missing descriptor/model")
    descriptor = read(execution / "descriptor.json")
    manifest = read(execution / "manifest.json")
    require(manifest["digest_alg"] == "sha256" and manifest["descriptor_digest"] ==
            "sha256:" + hashlib.sha256(canonical(descriptor)).hexdigest(), "Manifest descriptor binding differs")
    manifest_artifacts = {item["path"]: item for item in manifest["artifacts"]}
    require(len(manifest_artifacts) == len(manifest["artifacts"]) and set(manifest_artifacts) == set(closure_by_path), "Manifest closure differs")
    for name, item in manifest_artifacts.items():
        require(item["digest"] == "sha256:" + observed[name]["sha256"] and item["size"] == observed[name]["size_bytes"], "Manifest artifact differs")
    runtimes = [a["path"] for a in manifest["artifacts"] if a["role"] == "runtime"]
    require(runtimes == [report["artifacts_before"]["library_relative_path"]], "Runtime role differs")
    artifacts = report["artifacts_before"]
    require(artifacts["descriptor_raw_sha256"] == observed["descriptor.json"]["sha256"] and
            artifacts["descriptor_canonical_sha256"] == hashlib.sha256(canonical(descriptor)).hexdigest() and
            artifacts["manifest_sha256"] == observed["manifest.json"]["sha256"], "Descriptor/manifest identity differs")
    java_files = indexed(artifacts["java_files"])
    require({"bin/java", "release", "lib/modules", "lib/server/libjvm.so"} <= java_files.keys(), "Incomplete external JRE inventory")
    for boundary in ("before", "after"):
        board = report["board_" + boundary]
        require(board["system"] == "Linux" and board["machine"] == "aarch64", "Expected ARM64 Linux board")
    require(report["preflight"]["status"] == 0 and report["preflight"]["ok"] == 1 and
            report["open"]["status"] == report["close"]["status"] == 0, "Runtime lifecycle failed")
    durations = [report[name]["duration_ms"] for name in ("preflight", "open", "close")]
    require(all(type(value) in (float, int) and math.isfinite(value) and value >= 0 for value in durations), "Invalid lifecycle duration")
    require(type(supervisor["elapsed_seconds"]) in (float, int) and math.isfinite(supervisor["elapsed_seconds"]) and
            supervisor["elapsed_seconds"] > 0 and type(supervisor["timeout_seconds"]) in (float, int) and
            math.isfinite(supervisor["timeout_seconds"]) and supervisor["timeout_seconds"] > 0, "Invalid supervisor timing")
    require(len(report["warmups"]) == 3 and len(report["samples"]) == 30, "Incomplete workload")
    all_samples = report["warmups"] + report["samples"]
    previous_end = -1
    for sequence, sample in enumerate(all_samples):
        sample_metrics(sample, cases[ORDER[sequence % 3]], sequence, directory)
        require(sample["boundaries_ns"]["start"] >= previous_end, "Concurrent/overlapping samples")
        previous_end = sample["boundaries_ns"]["end"]
    elapsed_ms = supervisor["elapsed_seconds"] * 1000 + 1
    require(sum(durations) + sum(s["latency_ms"]["e2e"] for s in all_samples) <= elapsed_ms and
            (all_samples[-1]["boundaries_ns"]["end"] - all_samples[0]["boundaries_ns"]["start"]) / 1e6 <= elapsed_ms,
            "Sample/lifecycle timings exceed supervised process duration")
    aggregates = report["aggregates"]
    require(aggregates["percentile_method"] == "nearest_rank_ceil_p_times_n", "Percentile method differs")
    require(aggregates["all"] == stats(report["samples"]) and aggregates["by_case"] == {
        label: stats([s for s in report["samples"] if s["case_id"] == label]) for label in ORDER}, "Aggregate metrics differ")
    command = supervisor["command"]
    require(all(command.count(option) == 1 for option in ("--pack", "--library", "--workload")), "Missing/duplicate supervised argument")
    pack_argument = command[command.index("--pack") + 1]
    require(str(Path(pack_argument) / runtimes[0]) == command[command.index("--library") + 1] and
            str(Path(pack_argument) / PERFORMANCE["methodology"]) == command[command.index("--workload") + 1],
            "Supervised runtime/workload differs from audited pack")
    rss = check_rss(report, pack_argument, supervisor)
    return {"schema_version": 1, "status": "passed", "scope": "independent_native_performance_validation",
            "performance": PERFORMANCE, "benchmark_report": identity(directory / "benchmark.json"),
            "supervisor_report": identity(directory / "supervisor.json"), "workload_sha256": report["workload_sha256"],
            "execution_artifacts": closure, "metrics": {"aggregates": aggregates, "rss": rss,
                "open_ms": report["open"]["duration_ms"], "preflight_ms": report["preflight"]["duration_ms"]},
            "measured_sessions": 30, "warmup_sessions": 3, "sealed_cartridge": False, "full_spec_conformance": False,
            "limits": report["workload"]["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--expected-execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    try:
        result = check(args.benchmark_dir, args.expected_execution)
    except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
        result = {"schema_version": 1, "status": "failed", "reason": repr(error)}
    result["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["validator_sha256"] = sha_file(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
