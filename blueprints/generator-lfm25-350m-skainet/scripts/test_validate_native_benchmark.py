"""Synthetic evidence validation tests; the fixture never runs or times a model."""
import copy
import ctypes as C
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import benchmark_native_cartridge as bench
from test_benchmark_native_cartridge import FakeAbi, ROOT, WORKLOAD
import validate_native_benchmark as validator


def synthetic_evidence(root):
    """Produce consistent fake observations through the real sample serializer.

    Fake model/JAR/JRE bytes are used only as artifact-integrity test inputs. The
    generated text is fixture data, so this is never performance evidence.
    """
    directory, execution = root / "benchmark", root / "execution"
    directory.mkdir()
    execution.mkdir()
    work = bench.load_workload(WORKLOAD)
    samples = []
    ticks = iter(range(1_000_000, 1_000_000_000, 1000))
    for n, name in enumerate(bench.ORDER * 11):
        case = next(c for c in work["cases"] if c["id"] == name)
        phase = "warmup" if n < 3 else "measured"
        sample = bench.run_sample(FakeAbi(case), C.c_void_p(1), case, n, phase, clock=lambda: next(ticks))
        assert sample["status"] == "passed"
        samples.append(sample)
        stem = f"sample-{n:03d}-{phase}-{name}"
        bench.save(directory / (stem + ".json"), sample)
        bench.save(directory / (stem + ".started.json"), {"sequence": n, "phase": phase, "case_id": name,
                                                        "monotonic_ns": sample["boundaries_ns"]["start"] - 1})
    descriptor = {"id": "synthetic-test-only", "version": "0.1", "performance": validator.PERFORMANCE}
    files = {"descriptor.json": (json.dumps(descriptor, indent=2) + "\n").encode(),
             "artifacts/model/model.safetensors": b"synthetic weights; not model evidence",
             "artifacts/jvm/lib/worker.jar": b"synthetic jar", "artifacts/runtime/runtime.so": b"synthetic library",
             "artifacts/runtime/worker.json": b"{}", "artifacts/evaluation/native-abi-workload.json": WORKLOAD.read_bytes()}
    closure = []
    for name, content in files.items():
        path = execution / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        closure.append({"file": name, "source": name, **validator.identity(path)})
    manifest = {"id": descriptor["id"], "version": descriptor["version"], "digest_alg": "sha256",
                "descriptor_digest": "sha256:" + hashlib.sha256(bench.canonical(descriptor)).hexdigest(),
                "artifacts": [{"path": i["file"], "digest": "sha256:" + i["sha256"], "size": i["size_bytes"],
                               "role": "runtime" if i["file"].endswith("runtime.so") else "other"} for i in closure]}
    (execution / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    bench.save(execution / "execution.json", {"pack_files": closure})
    pack_files = [{key: item[key] for key in ("file", "sha256", "size_bytes")} for item in closure]
    pack_files.append({"file": "manifest.json", **validator.identity(execution / "manifest.json")})
    pack_files.append({"file": "execution.json", **validator.identity(execution / "execution.json")})
    java_files = [{"file": name, "sha256": hashlib.sha256(name.encode()).hexdigest(), "size_bytes": len(name)} for name in
                  ["bin/java", "release", "lib/modules", "lib/server/libjvm.so"] + [f"legal/file-{n}" for n in range(12)]]
    artifacts = {"pack_files": pack_files, "manifest_sha256": validator.sha_file(execution / "manifest.json"),
                 "descriptor_raw_sha256": validator.sha_file(execution / "descriptor.json"),
                 "descriptor_canonical_sha256": hashlib.sha256(bench.canonical(descriptor)).hexdigest(),
                 "java_files": java_files, "java_home": "/synthetic/java", "java_executable": "/synthetic/java/bin/java",
                 "library_relative_path": "artifacts/runtime/runtime.so"}
    command = ["python3", "benchmark_native_cartridge.py", "--_child", "--pack", "/synthetic/pack",
               "--library", "/synthetic/pack/artifacts/runtime/runtime.so", "--workload",
               "/synthetic/pack/artifacts/evaluation/native-abi-workload.json", "--output", "/synthetic/output"]
    consumer = {"pid": 100, "ppid": 99, "process_group": 100, "session_id": 100, "starttime_ticks": 10,
                "cmdline": command, "VmRSS_bytes": 1024, "VmHWM_bytes": 2048}
    worker = {"pid": 101, "ppid": 100, "process_group": 100, "session_id": 100, "starttime_ticks": 11,
              "cmdline": ["/synthetic/java/bin/java", bench.WORKER_MAIN, "--pack-dir", "/synthetic/pack"],
              "VmRSS_bytes": 4096, "VmHWM_bytes": 8192}
    rss_rows = [{"monotonic_ns": samples[0]["boundaries_ns"]["start"] - 100, "phase": "open_load",
                 "consumer": consumer, "workers": [worker]}]
    for sample in samples:
        rss_rows.append({"monotonic_ns": sample["boundaries_ns"]["start"],
                         "phase": f"{sample['phase']}-{sample['sequence']:03d}-{sample['case_id']}",
                         "consumer": consumer, "workers": [worker]})
    rss_rows.append({"monotonic_ns": samples[-1]["boundaries_ns"]["end"] + 1, "phase": "close",
                     "consumer": consumer, "workers": [worker]})
    rss = {"interval_ms": 100, "errors": [], "worker_identities": [worker], "samples": rss_rows,
           "peaks": {"consumer": {"VmRSS_bytes": 1024, "VmHWM_bytes": 2048},
                     "jvm": {"VmRSS_bytes": 4096, "VmHWM_bytes": 8192}}}
    report = {"schema_version": 1, "scope": "native_abi_short_chat_performance", "status": "passed", "errors": [],
              "performance_benchmark": True, "task_quality_qualified": False, "sealed_cartridge": False,
              "full_spec_conformance": False, "workload": work, "workload_sha256": bench.WORKLOAD_SHA256,
              "performance": validator.PERFORMANCE,
              "sources": [bench.file_record(ROOT / "scripts" / name, name) for name in
                          ("benchmark_native_cartridge.py", "consume_native_cartridge.py")],
              "artifacts_before": artifacts, "artifacts_after": copy.deepcopy(artifacts), "identities_unchanged": True,
              "board_before": {"system": "Linux", "machine": "aarch64"}, "board_after": {"system": "Linux", "machine": "aarch64"},
              "preflight": {"status": 0, "ok": 1, "duration_ms": 1.0}, "open": {"status": 0, "duration_ms": 2.0},
              "close": {"status": 0, "duration_ms": 1.0}, "warmups": samples[:3], "samples": samples[3:],
              "aggregates": bench.aggregate(samples[3:]), "rss": rss}
    bench.save(directory / "benchmark.json", report)
    (directory / "process.log").write_text("Synthetic test fixture, no inference executed.\n")
    supervisor = {"schema_version": 1, "scope": "native_abi_benchmark_owned_process", "status": "passed",
                  "exit_code": 0, "timed_out": False, "remaining_process_group_after_controller_exit": False,
                  "owned_process_group": 100, "command": command, "elapsed_seconds": 1.0, "timeout_seconds": 1800,
                  "benchmark_report": bench.file_record(directory / "benchmark.json", "benchmark.json"),
                  "log": bench.file_record(directory / "process.log", "process.log")}
    bench.save(directory / "supervisor.json", supervisor)
    return directory, execution


class NativeBenchmarkValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory, self.execution = synthetic_evidence(Path(self.temp.name))

    def mutate(self, operation, samples=False):
        report = json.loads((self.directory / "benchmark.json").read_bytes())
        operation(report)
        (self.directory / "benchmark.json").write_text(json.dumps(report))
        if samples:
            for sample in report["warmups"] + report["samples"]:
                name = f"sample-{sample['sequence']:03d}-{sample['phase']}-{sample['case_id']}.json"
                (self.directory / name).write_text(json.dumps(sample))
        supervisor = json.loads((self.directory / "supervisor.json").read_bytes())
        supervisor["benchmark_report"] = bench.file_record(self.directory / "benchmark.json", "benchmark.json")
        (self.directory / "supervisor.json").write_text(json.dumps(supervisor))

    def test_consistent_synthetic_evidence_passes(self):
        result = validator.check(self.directory, self.execution)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["measured_sessions"], 30)

    def test_modified_percentile_rejected(self):
        self.mutate(lambda r: r["aggregates"]["all"]["latency_ms"]["e2e"].update(p95=99))
        with self.assertRaisesRegex(ValueError, "Aggregate"):
            validator.check(self.directory, self.execution)

    def test_timestamp_diverging_from_pull_event_rejected(self):
        self.mutate(lambda r: r["samples"][0]["boundaries_ns"].update(first_text=r["samples"][0]["boundaries_ns"]["first_text"] + 1), samples=True)
        with self.assertRaisesRegex(ValueError, "Timing boundaries"):
            validator.check(self.directory, self.execution)

    def test_output_token_change_rejected_despite_matching_text(self):
        self.mutate(lambda r: r["samples"][0]["events"][-1]["payload"]["tokens"].__setitem__(0, 1234), samples=True)
        with self.assertRaisesRegex(ValueError, "Generated output"):
            validator.check(self.directory, self.execution)

    def test_source_change_rejected(self):
        self.mutate(lambda r: r["sources"][0].update(sha256="f" * 64))
        with self.assertRaisesRegex(ValueError, "Harness source"):
            validator.check(self.directory, self.execution)

    def test_supervisor_timeout_rejected(self):
        path = self.directory / "supervisor.json"
        report = json.loads(path.read_bytes())
        report["timed_out"] = True
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "Supervisor"):
            validator.check(self.directory, self.execution)

    def test_expected_execution_source_change_rejected(self):
        (self.execution / "artifacts/jvm/lib/worker.jar").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Expected execution source changed"):
            validator.check(self.directory, self.execution)

    def test_measured_closure_change_rejected(self):
        def change(report):
            for boundary in ("before", "after"):
                report["artifacts_" + boundary]["pack_files"][1]["sha256"] = "f" * 64
        self.mutate(change)
        with self.assertRaisesRegex(ValueError, "Measured execution differs"):
            validator.check(self.directory, self.execution)

    def test_extra_measured_classpath_jar_rejected(self):
        def add_jar(report):
            for boundary in ("before", "after"):
                report["artifacts_" + boundary]["pack_files"].append({"file": "artifacts/jvm/lib/extra.jar",
                    "sha256": hashlib.sha256(b"extra classpath code").hexdigest(), "size_bytes": 20})
        self.mutate(add_jar)
        with self.assertRaisesRegex(ValueError, "missing or extra files"):
            validator.check(self.directory, self.execution)

    def test_self_consistent_but_foreign_manifest_identity_rejected(self):
        def change(report):
            for boundary in ("before", "after"):
                artifacts = report["artifacts_" + boundary]
                artifacts["manifest_sha256"] = "f" * 64
                next(item for item in artifacts["pack_files"] if item["file"] == "manifest.json")["sha256"] = "f" * 64
        self.mutate(change)
        with self.assertRaisesRegex(ValueError, "Staging identity differs: manifest"):
            validator.check(self.directory, self.execution)

    def test_self_consistent_rss_from_foreign_process_group_rejected(self):
        def change(report):
            for row in report["rss"]["samples"]:
                row["consumer"].update(pid=200, process_group=200, session_id=200)
                for worker in row["workers"]:
                    worker.update(ppid=200, process_group=200, session_id=200)
            report["rss"]["worker_identities"][0].update(ppid=200, process_group=200, session_id=200)
        self.mutate(change)
        with self.assertRaisesRegex(ValueError, "not the supervised consumer"):
            validator.check(self.directory, self.execution)

    def test_supervised_library_outside_audited_pack_rejected(self):
        path = self.directory / "supervisor.json"
        supervisor = json.loads(path.read_bytes())
        command = supervisor["command"]
        command[command.index("--library") + 1] = "/different/lib.so"
        path.write_text(json.dumps(supervisor))
        self.mutate(lambda r: [row["consumer"].update(cmdline=command) for row in r["rss"]["samples"]])
        with self.assertRaisesRegex(ValueError, "runtime/workload differs"):
            validator.check(self.directory, self.execution)

    def test_negative_open_duration_rejected(self):
        self.mutate(lambda r: r["open"].update(duration_ms=-1))
        with self.assertRaisesRegex(ValueError, "Invalid lifecycle duration"):
            validator.check(self.directory, self.execution)

    def test_timing_span_exceeding_supervised_runtime_rejected(self):
        def add_delay(report):
            threshold = report["samples"][13]["boundaries_ns"]["start"]
            delay = 10_000_000_000  # Ten seconds cannot fit the one-second supervisor run.
            for sample in report["warmups"] + report["samples"]:
                if sample["boundaries_ns"]["start"] >= threshold:
                    sample["boundaries_ns"] = {key: value + delay for key, value in sample["boundaries_ns"].items()}
                    for record in sample["events"] + sample["abi_calls"]:
                        record["returned_monotonic_ns"] += delay
            for row in report["rss"]["samples"]:
                if row["monotonic_ns"] >= threshold:
                    row["monotonic_ns"] += delay
        self.mutate(add_delay, samples=True)
        with self.assertRaisesRegex(ValueError, "exceed supervised process duration"):
            validator.check(self.directory, self.execution)

    def test_absolute_expected_source_rejected_even_if_bytes_match(self):
        path = self.execution / "execution.json"
        expected = json.loads(path.read_bytes())
        expected["pack_files"][0]["source"] = str(self.execution / expected["pack_files"][0]["file"])
        path.write_text(json.dumps(expected))
        with self.assertRaisesRegex(ValueError, "identical safe relative path"):
            validator.check(self.directory, self.execution)


if __name__ == "__main__":
    unittest.main()
