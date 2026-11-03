"""No-model tests for exact-byte staging and the materialization acceptance gate."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("native_blueprint_prepare", Path(__file__).with_name("prepare.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        fake_pins = {}
        for name in m.MODEL_FILES:
            path = self.model / name
            path.write_bytes(("test fixture " + name).encode())
            fake_pins[name] = (path.stat().st_size, m.sha(path))
        pin_patch = patch.object(m, "MODEL_FILES", fake_pins)
        pin_patch.start()
        self.addCleanup(pin_patch.stop)
        self.raw = self.root / "descriptor.json"
        self.compact = self.root / "canonical.json"
        self.descriptor = {"id": "test-native-linux-aarch64", "version": "0.1.0-eval", "license": "NOASSERTION",
                           "performance": m.PERFORMANCE, "target": {"abi": "aarch64-linux-gnu"}}
        m.write_json(self.raw, self.descriptor)
        self.compact.write_bytes(m.canonical(self.descriptor))
        self.library = self.root / "test.so"
        self.library.write_bytes(b"test native artifact; no inference or ABI claim")
        self.build = self.root / "build.json"
        m.write_json(self.build, {"status": "built", "platform": {"system": "Linux", "machine": "aarch64"},
                                 "descriptor_sha256": m.sha(self.compact), "pack_descriptor_sha256": m.sha(self.raw),
                                 "output": m.identity(self.library), "static_libstdcxx": False,
                                 "sources": [{"file": "bridge.cpp", "sha256": "test-fixture"}]})
        self.jars = self.root / "lib"
        self.jars.mkdir()
        with zipfile.ZipFile(self.jars / "worker.jar", "w") as jar:
            jar.writestr("sk/ainet/cartridge/generator/lfm2/worker/MainKt.class", b"test fixture")
        self.compiler = self.root / "compiler.json"
        m.write_json(self.compiler, {"test": "fixture"})
        self.license = self.root / "LICENSE"
        self.license.write_text("test fixture")
        self.workload = self.root / "workload.json"
        m.write_json(self.workload, {"test": "fixture"})

    def prepare(self, name="execution"):
        target = self.root / name
        m.prepare(argparse.Namespace(output=target, descriptor=self.raw, canonical_descriptor=self.compact,
                                    native_library=self.library, native_build_report=self.build,
                                    compiler_report=self.compiler, native_license=self.license,
                                    model_dir=self.model, worker_libs=self.jars, workload=self.workload))
        return target

    def accepted_inputs(self, execution):
        benchmark = self.root / "benchmark"
        benchmark.mkdir()
        m.write_json(benchmark / "benchmark.json", {"raw": "test evidence"})
        m.write_json(benchmark / "supervisor.json", {"raw": "test supervision"})
        validation = self.root / "validation.json"
        m.write_json(validation, {"schema_version": 1, "status": "passed", "performance": m.PERFORMANCE,
                                  "benchmark_report": m.identity(benchmark / "benchmark.json"),
                                  "supervisor_report": m.identity(benchmark / "supervisor.json"),
                                  "workload_sha256": m.sha(execution / m.PERFORMANCE["methodology"]),
                                  "execution_artifacts": [{k: v for k, v in item.items() if k != "source"}
                                                          for item in m.closure(execution)]})
        preview = self.root / "preview.json"
        m.write_json(preview, {"blueprint": {"id": "test"}, "target": "linux-aarch64",
                              "measurements": {"performance": m.PERFORMANCE, "quality": m.QUALITY}})
        return argparse.Namespace(validation=validation, benchmark_dir=benchmark, execution=execution,
                                  preview_profile=preview, output=self.root / "accepted.json")

    def test_raw_and_canonical_descriptor_must_identify_exact_same_object(self):
        self.raw.write_text(json.dumps({**self.descriptor, "version": "changed"}))
        with self.assertRaisesRegex(ValueError, "Canonical ABI descriptor"):
            m.descriptor_pair(self.raw, self.compact)

    def test_existing_failed_directory_is_never_reused(self):
        path = self.root / "failed"
        m.claim_directory(path)
        with self.assertRaises(FileExistsError):
            m.claim_directory(path)
        self.assertTrue((path / ".run-started").is_file())

    def test_staging_reconstructs_exact_closure_with_local_model(self):
        execution = self.prepare()
        staged = self.root / "device-pack"
        m.stage(argparse.Namespace(execution=execution, model_dir=self.model, output=staged))
        self.assertEqual(m.read(execution / "execution.json"), m.read(staged / "execution.json"))
        self.assertEqual(m.read(staged / "manifest.json")["digest_alg"], "sha256")
        self.assertFalse((staged / "artifacts/runtime/java").exists())

    def test_later_build_output_overwrite_cannot_mutate_staged_jar(self):
        execution = self.prepare()
        staged = execution / "artifacts/jvm/lib/worker.jar"
        original = staged.read_bytes()
        (self.jars / "worker.jar").write_bytes(b"new installDist output")
        self.assertEqual(staged.read_bytes(), original)
        record = next(item for item in m.read(execution / "execution.json")["pack_files"]
                      if item["file"] == "artifacts/jvm/lib/worker.jar")
        m.verify(staged, record)

    def test_changed_transported_jar_fails_and_preserves_attempt(self):
        execution = self.prepare()
        (execution / "artifacts/jvm/lib/worker.jar").write_bytes(b"changed")
        staged = self.root / "bad-device-pack"
        with self.assertRaisesRegex(ValueError, "Artifact identity mismatch"):
            m.stage(argparse.Namespace(execution=execution, model_dir=self.model, output=staged))
        self.assertTrue((staged / ".run-started").is_file())
        self.assertFalse((staged / "execution.json").exists())

    def test_unsafe_transport_path_is_rejected(self):
        execution = self.prepare()
        record = m.read(execution / "execution.json")
        record["pack_files"][0]["file"] = "../escape"
        (execution / "execution.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Unsafe artifact path"):
            m.stage(argparse.Namespace(execution=execution, model_dir=self.model, output=self.root / "unsafe"))

    def test_acceptance_binds_raw_report_bytes_and_exact_execution_closure(self):
        execution = self.prepare()
        args = self.accepted_inputs(execution)
        (args.benchmark_dir / "benchmark.json").write_text('{"raw":"changed"}')
        with self.assertRaisesRegex(ValueError, "Accepted evidence digest mismatch"):
            m.accepted(args)
        self.assertFalse(args.output.exists())

    def test_accepted_profile_has_only_accepted_performance_and_digest_pinned_reports(self):
        execution = self.prepare()
        args = self.accepted_inputs(execution)
        m.accepted(args)
        result = m.read(args.output)
        self.assertEqual(result["measurements"], {"performance": m.PERFORMANCE, "quality": m.QUALITY})
        self.assertEqual(set(result["inputs"]), set(m.REPORT_PATHS))
        self.assertNotIn("signing", result)
        self.assertNotIn("license_acceptance", result)
        self.assertEqual(result["inputs"]["native-benchmark"]["digest"], "sha256:" + m.sha(args.benchmark_dir / "benchmark.json"))

    def test_added_execution_file_invalidates_acceptance(self):
        execution = self.prepare()
        args = self.accepted_inputs(execution)
        (execution / "artifacts/runtime/unmeasured.cfg").write_text("changed closure")
        with self.assertRaisesRegex(ValueError, "Accepted execution closure differs"):
            m.accepted(args)

    def test_rehashing_changed_final_report_cannot_bypass_accepted_input_pin(self):
        execution = self.prepare()
        accepted_args = self.accepted_inputs(execution)
        m.accepted(accepted_args)
        pack = self.root / "materialized"
        shutil.copytree(execution, pack)
        accepted = m.read(accepted_args.output)
        for name, relative in m.REPORT_PATHS.items():
            shutil.copyfile(accepted["inputs"][name]["path"], pack / relative)

        def refresh_manifest():
            mfest = {"effective_license": "NOASSERTION", "descriptor_digest": "sha256:" + m.sha(execution / m.CANONICAL),
                     "artifacts": [{"path": item["file"], "role": m.role(item["file"]),
                                    "digest": "sha256:" + item["sha256"], "size": item["size_bytes"]}
                                   for item in m.closure(pack)]}
            (pack / "manifest.json").write_text(json.dumps(mfest))

        refresh_manifest()
        m.verify_materialized(argparse.Namespace(execution=execution, pack=pack,
                                                accepted_profile=accepted_args.output, output=self.root / "verified.json"))
        (pack / m.REPORT_PATHS["native-benchmark"]).write_text('{"raw":"changed and rehashed"}')
        refresh_manifest()
        with self.assertRaisesRegex(ValueError, "Artifact identity mismatch"):
            m.verify_materialized(argparse.Namespace(execution=execution, pack=pack,
                                                    accepted_profile=accepted_args.output, output=self.root / "rehashed.json"))


if __name__ == "__main__":
    unittest.main()
