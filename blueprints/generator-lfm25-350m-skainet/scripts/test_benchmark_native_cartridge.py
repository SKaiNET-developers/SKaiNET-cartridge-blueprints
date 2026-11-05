"""Focused benchmark harness tests; no native library, model or device execution."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import benchmark_native_cartridge as bench
from validate_native_cartridge import references

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "samples/native-abi-workload.json"


class FakeAbi:
    """Exercise real ctypes pointer ownership/decoding without model execution."""
    def __init__(self, case, failure=None, wrong_token=False):
        expected = case["expected"]
        final = {"type": "final", "text": expected["text"], "tokens": list(expected["tokens"]),
                 "finish_reason": expected["finish_reason"]}
        if wrong_token:
            final["tokens"][0] += 1
        self.events = [(1, {"type": "ready", "prompt_ids": expected["prompt_ids"]}),
                       (1, {"type": "delta", "text": expected["text"]}), (2, final)]
        self.failure, self.results, self.released = failure, {}, []
        self.ended, self.diagnostics = False, []
        self.pushed = None

    def ctg_session_begin(self, cartridge, opts, out):
        out._obj.value = 42
        return 0

    def ctg_push(self, session, view):
        self.pushed = C.string_at(view._obj.data, view._obj.len)
        return -1 if self.failure == "push" else 0

    def ctg_flush(self, session):
        return 0

    def ctg_pull(self, session, out):
        if self.failure == "pull":
            return -5
        if self.events:
            kind, document = self.events.pop(0)
            number = len(self.released) + 100
            self.results[number] = (kind, C.create_string_buffer(json.dumps(document).encode()))
            out._obj.value = number
        return 0

    def ctg_result_get_payload(self, result, view):
        memory = self.results[result.value][1]
        view._obj.data, view._obj.len = C.cast(memory, C.c_void_p), len(memory.value)
        return 0

    def ctg_result_get_kind(self, result):
        return self.results[result.value][0]

    def ctg_result_release(self, result):
        self.released.append(result.value)
        self.results.pop(result.value)

    def ctg_session_end(self, session):
        self.ended = True
        return -5 if self.failure == "end" else 0

    def ctg_last_error(self, handle):
        self.diagnostics.append(handle)
        return b"synthetic ABI failure"


class BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = bench.load_workload(WORKLOAD)

    def test_workload_exactly_matches_independent_reference_traces(self):
        for case in self.work["cases"]:
            ref = references()[case["id"]]
            self.assertEqual(case["request"], {"messages": ref["messages"], "max_output_tokens": ref["max_new_tokens"]})
            self.assertEqual(case["expected"], {"prompt_ids": ref["prompt_ids"], "tokens": ref["expected_generated_ids"],
                             "text": ref["expected_decoded_completion_skip_special_tokens"], "finish_reason": "STOP"})
        self.assertEqual(self.work["warmup_order"], bench.ORDER)
        self.assertEqual(self.work["measured_order"], bench.ORDER * 10)

    def test_sample_owns_results_and_records_native_boundary_times(self):
        case = self.work["cases"][0]
        fake = FakeAbi(case)
        ticks = iter(range(1000, 100000, 100))
        sample = bench.run_sample(fake, C.c_void_p(1), case, 3, "measured", clock=lambda: next(ticks))
        self.assertEqual(sample["status"], "passed")
        self.assertTrue(fake.ended)
        self.assertEqual(len(fake.released), 3)
        self.assertFalse(fake.results)
        self.assertEqual(json.loads(fake.pushed), case["request"])
        boundaries = sample["boundaries_ns"]
        self.assertEqual(boundaries["first_text"], sample["events"][1]["returned_monotonic_ns"])
        self.assertEqual(sample["latency_ms"]["e2e"], (boundaries["end"] - boundaries["start"]) / 1e6)
        self.assertGreater(sample["latency_ms"]["e2e"], sample["latency_ms"]["final"])
        self.assertEqual(sample["generated_tokens"], 8)

    def test_failed_abi_call_preserves_status_and_cleans_session(self):
        for name, status in (("push", -1), ("pull", -5), ("end", -5)):
            with self.subTest(name=name):
                case = self.work["cases"][0]
                fake = FakeAbi(case, failure=name)
                sample = bench.run_sample(fake, C.c_void_p(1), case, 3, "measured")
                self.assertEqual(sample["status"], "failed")
                self.assertTrue(fake.ended)
                self.assertTrue(any(c["status"] == status and c["detail"] for c in sample["abi_calls"]))
                self.assertNotIn("latency_ms", sample)
                if name == "end":
                    self.assertEqual(fake.diagnostics, [None], "Never query consumed session handle")

    def test_matching_text_with_wrong_token_fails_fidelity(self):
        case = self.work["cases"][0]
        sample = bench.run_sample(FakeAbi(case, wrong_token=True), C.c_void_p(1), case, 3, "measured")
        self.assertEqual(sample["status"], "failed")
        self.assertIn("Final text, tokens or reason", sample["error"])

    def test_nearest_rank_and_equal_mix_aggregation(self):
        samples = []
        for n, name in enumerate(bench.ORDER * 10, 1):
            samples.append({"case_id": name, "status": "passed", "prompt_tokens": 2, "generated_tokens": 1,
                            "latency_ms": {m: n for m in ("e2e", "ready", "ttft", "final", "drain")}})
        stats = bench.aggregate(samples)
        self.assertEqual(stats["all"]["latency_ms"]["e2e"], {"p50": 15, "p95": 29})
        self.assertEqual(stats["by_case"]["arithmetic"]["latency_ms"]["e2e"], {"p50": 13, "p95": 28})
        with self.assertRaises(ValueError):
            bench.aggregate(samples[:-1])
        samples[-1]["status"] = "failed"
        with self.assertRaises(ValueError):
            bench.aggregate(samples)

    def test_proc_identity_handles_spaces_parentheses_and_checks_worker_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp)
            p = proc / "123"
            p.mkdir()
            fields = ["S", "77", "77", "77"] + ["0"] * 15 + ["9999"]
            (p / "stat").write_text("123 (java (worker)) " + " ".join(fields))
            (p / "status").write_text("Name:\tjava\nVmRSS:\t123 kB\nVmHWM:\t200 kB\n")
            (p / "cmdline").write_bytes(b"java\0" + bench.WORKER_MAIN.encode() + b"\0--pack-dir\0/pack\0")
            observed = bench.proc_record(123, proc)
            self.assertEqual(observed["starttime_ticks"], 9999)
            self.assertEqual(observed["VmRSS_bytes"], 123 * 1024)
            self.assertTrue(bench.is_worker(observed, 77, Path("/pack")))
            self.assertFalse(bench.is_worker(observed, 78, Path("/pack")))
            self.assertFalse(bench.is_worker(observed, 77, Path("/other")))


class ArtifactAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack, self.jre = self.root / "pack", self.root / "jre"
        for relative in ("bin/java", "release", "lib/modules", "lib/server/libjvm.so"):
            path = self.jre / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
        (self.jre / "bin/java").chmod(0o700)
        files = {"artifacts/model/model.safetensors": b"weights", "artifacts/model/config.json": b"{}",
                 "artifacts/jvm/lib/worker.jar": b"jar", "artifacts/runtime/runtime.so": b"native"}
        for relative, data in files.items():
            path = self.pack / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        config = {"model_dir": "artifacts/model", "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(),
                  "metadata_sha256": {"config.json": hashlib.sha256(b"{}").hexdigest()},
                  "decoder_policy": "HUGGING_FACE_5_COMPAT", "max_output_tokens": 256}
        (self.pack / "artifacts/runtime/worker.json").write_text(json.dumps(config))
        self.work = {"model": config, "decoder_policy": config["decoder_policy"]}
        self.descriptor = {"id": "fake", "kind": "cartridge"}
        (self.pack / "descriptor.json").write_text(json.dumps(self.descriptor, indent=2) + "\n")
        self.library = self.pack / "artifacts/runtime/runtime.so"
        artifacts = [{"path": r["file"], "digest": "sha256:" + r["sha256"], "size": r["size_bytes"],
                      "role": "runtime" if r["file"].endswith("runtime.so") else "other"}
                     for r in bench.tree_inventory(self.pack)]
        self.manifest = {"artifacts": artifacts, "digest_alg": "sha256",
                         "descriptor_digest": "sha256:" + hashlib.sha256(bench.canonical(self.descriptor)).hexdigest()}
        (self.pack / "manifest.json").write_text(json.dumps(self.manifest))

    def test_pretty_descriptor_canonical_digest_and_full_jre_closure(self):
        with patch.dict(os.environ, {"JAVA_HOME": str(self.jre)}):
            inventory = bench.audit_artifacts(self.pack, self.library, self.work)
        self.assertNotEqual(inventory["descriptor_raw_sha256"], inventory["descriptor_canonical_sha256"])
        self.assertEqual(inventory["descriptor_canonical_sha256"], self.manifest["descriptor_digest"][7:])
        self.assertEqual({f["file"] for f in inventory["java_files"]}, {"bin/java", "release", "lib/modules", "lib/server/libjvm.so"})

    def test_raw_descriptor_digest_is_not_accepted_as_canonical_identity(self):
        self.manifest["descriptor_digest"] = "sha256:" + bench.sha_file(self.pack / "descriptor.json")
        (self.pack / "manifest.json").write_text(json.dumps(self.manifest))
        with patch.dict(os.environ, {"JAVA_HOME": str(self.jre)}), self.assertRaisesRegex(ValueError, "Canonical descriptor"):
            bench.audit_artifacts(self.pack, self.library, self.work)

    def test_manifest_artifact_mutation_is_rejected(self):
        self.library.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "Manifest artifact differs"):
            bench.audit_artifacts(self.pack, self.library, self.work)

    def test_broken_or_nonexecutable_packaged_java_does_not_fall_back(self):
        packaged = self.pack / "artifacts/runtime/java"
        with patch.dict(os.environ, {"JAVA_HOME": str(self.jre)}):
            packaged.symlink_to(self.root / "missing-java")
            with self.assertRaisesRegex(ValueError, "refusing fallback"):
                bench.java_home_for(self.pack)
            packaged.unlink()
            packaged.write_text("not executable")
            packaged.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "refusing fallback"):
                bench.java_home_for(self.pack)
            packaged.unlink()
            packaged.symlink_to(self.jre / "bin/java")
            self.assertEqual(bench.java_home_for(self.pack), ((self.jre / "bin/java").resolve(), self.jre.resolve()))


class SupervisorTest(unittest.TestCase):
    def test_spawn_failure_still_has_exclusive_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            report = bench.supervise([str(output / "no-such-executable")], output, 1)
            self.assertEqual(report["status"], "failed")
            self.assertIsNone(report["exit_code"])
            self.assertIn("error", report)
            self.assertTrue((output / "supervisor.json").is_file())

    def test_timeout_preserves_partial_sample_and_only_terminates_owned_group(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
            try:
                code = "from pathlib import Path; import time; Path('sample-partial.json').write_text('{}'); time.sleep(10)"
                # Use an explicit absolute output path; no shell or process-name kills.
                code = code.replace("'sample-partial.json'", repr(str(output / "sample-partial.json")))
                report = bench.supervise([sys.executable, "-c", code], output, .2)
                self.assertTrue(report["timed_out"])
                self.assertEqual(report["status"], "failed")
                self.assertTrue((output / "sample-partial.json").is_file())
                self.assertIsNone(unrelated.poll())
                self.assertTrue((output / "supervisor.json").is_file())
                self.assertLess(report["elapsed_seconds"], 8)
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=5)

    def test_exclusive_report_write_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            bench.save(path, {"first": True})
            with self.assertRaises(FileExistsError):
                bench.save(path, {"second": True})
            self.assertEqual(json.loads(path.read_bytes()), {"first": True})


if __name__ == "__main__":
    unittest.main()
