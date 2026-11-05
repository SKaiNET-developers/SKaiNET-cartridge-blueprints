import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class Buffer(C.Structure):
    _fields_ = [("size", C.c_size_t), ("data", C.c_void_p), ("len", C.c_size_t)]


class Options(C.Structure):
    _fields_ = [("size", C.c_size_t), ("flags", C.c_uint32)]


class Preflight(C.Structure):
    _fields_ = [("size", C.c_size_t), ("ok", C.c_int32), ("detail", C.c_char_p)]


class NativeBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.build = Path(cls.temporary.name)
        cls.descriptor = json.dumps({"id": "abi-fake-test", "io": {"mode": "streaming"}, "evaluation_only": True},
                                   sort_keys=True, separators=(",", ":")).encode()
        source = cls.build / "descriptor.json"
        source.write_bytes(cls.descriptor)
        suffix = ".dylib" if platform.system() == "Darwin" else ".so"
        library = cls.build / ("liblfm2_cartridge" + suffix)
        subprocess.run([sys.executable, str(ROOT / "build.py"), "--descriptor", str(source), "--output", str(library)],
                       check=True, capture_output=True, text=True, timeout=40)
        cls.lib = C.CDLL(str(library))
        cls.library = library
        cls.pack_descriptor = json.dumps(json.loads(cls.descriptor), indent=2).encode() + b"\n"
        raw_source = cls.build / "materialized-descriptor.json"
        raw_source.write_bytes(cls.pack_descriptor)
        raw_library = cls.build / ("liblfm2_materialized" + suffix)
        subprocess.run([sys.executable, str(ROOT / "build.py"), "--descriptor", str(source),
                        "--pack-descriptor", str(raw_source), "--output", str(raw_library)],
                       check=True, capture_output=True, text=True, timeout=40)
        cls.raw_lib = C.CDLL(str(raw_library))
        cls.raw_lib.ctg_descriptor_json.restype = C.c_char_p
        cls.raw_lib.ctg_preflight.argtypes = [C.c_char_p, C.POINTER(Preflight)]
        cls.raw_lib.ctg_preflight.restype = C.c_int
        cls.raw_build_report = json.loads(raw_library.with_name(raw_library.name + ".build.json").read_text())
        cls.lib.ctg_abi_version.restype = C.c_uint32
        cls.lib.ctg_descriptor_json.restype = C.c_char_p
        signatures = {
            "ctg_preflight": ([C.c_char_p, C.POINTER(Preflight)], C.c_int),
            "ctg_open": ([C.c_char_p, C.POINTER(Options), C.POINTER(C.c_void_p)], C.c_int),
            "ctg_close": ([C.c_void_p], C.c_int),
            "ctg_session_begin": ([C.c_void_p, C.POINTER(Options), C.POINTER(C.c_void_p)], C.c_int),
            "ctg_session_end": ([C.c_void_p], C.c_int),
            "ctg_push": ([C.c_void_p, C.POINTER(Buffer)], C.c_int),
            "ctg_flush": ([C.c_void_p], C.c_int),
            "ctg_pull": ([C.c_void_p, C.POINTER(C.c_void_p)], C.c_int),
            "ctg_cancel": ([C.c_void_p], C.c_int),
            "ctg_infer": ([C.c_void_p, C.POINTER(Buffer), C.POINTER(C.c_void_p)], C.c_int),
            "ctg_result_get_kind": ([C.c_void_p], C.c_int),
            "ctg_result_get_payload": ([C.c_void_p, C.POINTER(Buffer)], C.c_int),
            "ctg_result_release": ([C.c_void_p], None),
            "ctg_last_error": ([C.c_void_p], C.c_void_p),
        }
        for name, (args, result) in signatures.items():
            function = getattr(cls.lib, name)
            function.argtypes = args
            function.restype = result

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.pack = Path(self.directory.name)
        (self.pack / "artifacts/runtime").mkdir(parents=True)
        (self.pack / "artifacts/jvm/lib").mkdir(parents=True)
        (self.pack / "descriptor.json").write_bytes(self.descriptor)
        java = self.pack / "artifacts/runtime/java"
        java.write_text((ROOT / "tests/fake_java.py").read_text())
        java.chmod(0o755)
        self.mode("normal")

    def mode(self, value):
        (self.pack / "artifacts/runtime/worker.json").write_text(json.dumps({"fake_mode": value}))

    def open(self):
        handle = C.c_void_p()
        self.assertEqual(0, self.lib.ctg_open(os.fsencode(self.pack), None, C.byref(handle)), self.error(None))
        self.assertTrue(handle.value)
        self.addCleanup(lambda: self.close(handle) if handle.value else None)
        return handle

    def close(self, handle):
        code = self.lib.ctg_close(handle)
        if code == 0: handle.value = None
        return code

    def session(self, cartridge):
        handle = C.c_void_p()
        self.assertEqual(0, self.lib.ctg_session_begin(cartridge, None, C.byref(handle)))
        self.addCleanup(lambda: self.end(handle) if handle.value else None)
        return handle

    def end(self, handle):
        code = self.lib.ctg_session_end(handle)
        handle.value = None
        return code

    def error(self, handle):
        return C.string_at(self.lib.ctg_last_error(handle)).decode()

    def push(self, session, data):
        raw = C.create_string_buffer(data)
        view = Buffer(C.sizeof(Buffer), C.cast(raw, C.c_void_p), len(data))
        code = self.lib.ctg_push(session, C.byref(view))
        C.memset(raw, ord("X"), len(data))
        return code

    def payload(self, result):
        view = Buffer(C.sizeof(Buffer), None, 0)
        self.assertEqual(0, self.lib.ctg_result_get_payload(result, C.byref(view)))
        return C.string_at(view.data, view.len)

    def pull(self, session):
        result = C.c_void_p()
        code = self.lib.ctg_pull(session, C.byref(result))
        return code, result

    def drain(self, session):
        events = []
        while True:
            code, result = self.pull(session)
            self.assertEqual(0, code, self.error(session))
            if not result.value: return events
            events.append((self.lib.ctg_result_get_kind(result), json.loads(self.payload(result))))
            self.lib.ctg_result_release(result)

    def assert_reaped(self):
        pid = int((self.pack / "fake-worker.pid").read_text())
        with self.assertRaises(ProcessLookupError): os.kill(pid, 0)

    def test_identity_preflight_versions_and_descriptor_mismatch_without_model(self):
        self.assertEqual(1, self.lib.ctg_abi_version())
        self.assertEqual(self.descriptor, self.lib.ctg_descriptor_json())
        report = Preflight(C.sizeof(Preflight), 0, None)
        self.assertEqual(0, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
        self.assertEqual(1, report.ok)
        self.assertFalse((self.pack / "fake-worker.pid").exists())
        self.mode("wrong_java")
        self.assertEqual(-3, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
        self.assertEqual(0, report.ok)
        self.mode("normal")
        (self.pack / "descriptor.json").write_bytes(self.descriptor + b"\n")
        self.assertEqual(-3, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))

    def test_backend_preflight_rejects_failed_or_ambiguous_readiness_before_open(self):
        report = Preflight(C.sizeof(Preflight), 0, None)
        for mode in ("preflight_error", "preflight_invalid", "preflight_nested", "preflight_duplicate",
                     "preflight_trailing", "preflight_unknown_backend", "preflight_model_loaded", "preflight_model_missing"):
            with self.subTest(mode=mode):
                self.mode(mode)
                self.assertEqual(-3, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
                self.assertEqual(0, report.ok)
                self.assertFalse((self.pack / "fake-worker.pid").exists())
                pid = int((self.pack / "fake-preflight.pid").read_text())
                with self.assertRaises(ProcessLookupError): os.kill(pid, 0)
        self.mode("preflight_error")
        handle = C.c_void_p()
        self.assertEqual(-3, self.lib.ctg_open(os.fsencode(self.pack), None, C.byref(handle)))
        self.assertFalse(handle.value)
        self.assertFalse((self.pack / "fake-worker.pid").exists())

    def test_backend_preflight_accepts_gpu_readiness_without_starting_generation(self):
        self.mode("preflight_gpu")
        report = Preflight(C.sizeof(Preflight), 0, None)
        self.assertEqual(0, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
        self.assertEqual(1, report.ok)
        self.assertIn(b"backend requirements ready", report.detail)
        self.assertFalse((self.pack / "fake-worker.pid").exists())
        pid = int((self.pack / "fake-preflight.pid").read_text())
        with self.assertRaises(ProcessLookupError): os.kill(pid, 0)

    def test_backend_preflight_output_and_timeout_bounds_reap_owned_child(self):
        report = Preflight(C.sizeof(Preflight), 0, None)
        for mode, bound in (("preflight_oversized", 4), ("preflight_hang", 34)):
            with self.subTest(mode=mode):
                self.mode(mode)
                start = time.monotonic()
                self.assertEqual(-3, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
                self.assertLess(time.monotonic() - start, bound)
                self.assertEqual(0, report.ok)
                pid = int((self.pack / "fake-preflight.pid").read_text())
                with self.assertRaises(ProcessLookupError): os.kill(pid, 0)
                nested = int((self.pack / "fake-nested-preflight.pid").read_text())
                with self.assertRaises(ProcessLookupError): os.kill(nested, 0)
        self.assertFalse((self.pack / "fake-worker.pid").exists())

    def test_materialized_descriptor_has_canonical_abi_and_exact_file_identity(self):
        self.assertEqual(self.descriptor, self.raw_lib.ctg_descriptor_json())
        report = Preflight(C.sizeof(Preflight), 0, None)
        self.assertEqual(-3, self.raw_lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
        (self.pack / "descriptor.json").write_bytes(self.pack_descriptor)
        self.assertEqual(0, self.raw_lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
        self.assertEqual(1, report.ok)
        self.assertFalse((self.pack / "fake-worker.pid").exists())
        (self.pack / "descriptor.json").write_bytes(self.pack_descriptor.replace(b"abi-fake-test", b"different-id"))
        self.assertEqual(-3, self.raw_lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
        for name, data in (("descriptor", self.descriptor), ("pack_descriptor", self.pack_descriptor)):
            expected = hashlib.sha256(data).hexdigest()
            self.assertEqual(expected, self.raw_build_report[name + "_sha256"])
            self.assertEqual(expected, self.raw_build_report[name + "_input"]["sha256"])
            self.assertEqual(len(data), self.raw_build_report[name + "_input"]["size_bytes"])

    @unittest.skipUnless(platform.system() == "Darwin", "Darwin install-name contract")
    def test_darwin_runtime_install_name_is_relocatable(self):
        result = subprocess.run(["otool", "-D", str(self.library)], check=True,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual([str(self.library) + ":", "@rpath/" + self.library.name], result.stdout.splitlines())

    def test_builder_rejects_different_or_ambiguous_pack_identity(self):
        raw = self.pack / "raw-descriptor.json"
        output = self.pack / self.library.name
        for data in (b'{"id":"different"}',
                     self.descriptor[:-1] + b',"id":"abi-fake-test"}',
                     self.descriptor + b"\0"):
            with self.subTest(data=data):
                raw.write_bytes(data)
                result = subprocess.run([sys.executable, str(ROOT / "build.py"),
                                         "--descriptor", str(self.build / "descriptor.json"),
                                         "--pack-descriptor", str(raw), "--output", str(output)],
                                        capture_output=True, timeout=10)
                self.assertNotEqual(0, result.returncode)
                self.assertFalse(output.exists())
                self.assertFalse(output.with_name(output.name + ".build.json").exists())

    def test_external_java_home_is_required_and_validated_before_open(self):
        packaged = self.pack / "artifacts/runtime/java"
        home = self.pack / "external-jre"
        (home / "bin").mkdir(parents=True)
        external = home / "bin/java"
        external.write_bytes(packaged.read_bytes())
        external.chmod(0o755)
        external_config = home / "bin/worker.json"
        external_config.write_text('{"fake_mode":"normal"}')
        packaged.unlink()
        report = Preflight(C.sizeof(Preflight), 0, None)
        with mock.patch.dict(os.environ, {"JAVA_HOME": ""}):
            self.assertEqual(-3, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
            self.assertIn(b"JAVA_HOME", report.detail)
        with mock.patch.dict(os.environ, {"JAVA_HOME": str(home / "missing")}):
            self.assertEqual(-3, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
        with mock.patch.dict(os.environ, {"JAVA_HOME": str(home)}):
            external_config.write_text('{"fake_mode":"wrong_java"}')
            self.assertEqual(-3, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
            self.assertEqual(0, report.ok)
            external_config.write_text('{"fake_mode":"normal"}')
            self.assertEqual(0, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
            self.assertFalse((self.pack / "fake-worker.pid").exists())
            cartridge = self.open()
            session = self.session(cartridge)
            self.assertEqual(0, self.push(session, b'{"reply":"external"}'))
            self.assertEqual(0, self.lib.ctg_flush(session))
            self.assertEqual("external", self.drain(session)[-1][1]["text"])
            self.assertEqual(0, self.end(session))
            self.assertEqual(0, self.close(cartridge))
            self.assert_reaped()

    def test_package_java_precedes_home_and_invalid_package_does_not_fall_back(self):
        report = Preflight(C.sizeof(Preflight), 0, None)
        with mock.patch.dict(os.environ, {"JAVA_HOME": str(self.pack / "not-a-jre")}):
            self.assertEqual(0, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
        home = self.pack / "external-jre"
        (home / "bin").mkdir(parents=True)
        packaged = self.pack / "artifacts/runtime/java"
        external = home / "bin/java"
        external.write_bytes(packaged.read_bytes())
        external.chmod(0o755)
        (home / "bin/worker.json").write_text('{"fake_mode":"normal"}')
        with mock.patch.dict(os.environ, {"JAVA_HOME": str(home)}):
            packaged.chmod(0o644)
            self.assertEqual(-3, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
            packaged.unlink()
            packaged.symlink_to(self.pack / "missing-java")
            self.assertEqual(-3, self.lib.ctg_preflight(os.fsencode(self.pack), C.byref(report)))
        self.assertFalse((self.pack / "fake-worker.pid").exists())

    def test_open_reuses_resolved_java_when_probe_retargets_package_symlink(self):
        packaged = self.pack / "artifacts/runtime/java"
        selected = packaged.with_name("selected-java")
        packaged.rename(selected)
        packaged.symlink_to(selected)
        replacement = packaged.with_name("invalid-java")
        replacement.write_text("#!/bin/sh\nexit 97\n")
        replacement.chmod(0o755)
        (self.pack / "artifacts/runtime/worker.json").write_text(json.dumps({
            "fake_mode": "normal", "replace_java_link": str(packaged), "replace_java_target": str(replacement)}))
        cartridge = self.open()
        self.assertEqual(replacement.resolve(), packaged.resolve())
        self.assertEqual(0, self.close(cartridge))
        self.assert_reaped()

    def test_struct_size_flags_and_success_preserve_last_error(self):
        output = C.c_void_p()
        small = Options(C.sizeof(Options) - 1, 0)
        self.assertEqual(-8, self.lib.ctg_open(os.fsencode(self.pack), C.byref(small), C.byref(output)))
        self.assertFalse(output.value)
        bad = Options(C.sizeof(Options), 1)
        self.assertEqual(-1, self.lib.ctg_open(os.fsencode(self.pack), C.byref(bad), C.byref(output)))
        cartridge = self.open()
        session = self.session(cartridge)
        self.assertEqual(-8, self.lib.ctg_push(session, C.byref(Buffer(0, None, 0))))
        self.assertEqual(-2, self.lib.ctg_infer(session, None, C.byref(output)))
        address = self.lib.ctg_last_error(session)
        original = C.string_at(address)
        self.assertEqual(0, self.push(session, b'{"reply":"stable"}'))
        self.assertEqual(0, self.lib.ctg_flush(session))
        self.drain(session)
        self.assertEqual(address, self.lib.ctg_last_error(session))
        self.assertEqual(original, C.string_at(address))
        self.assertEqual(-7, self.lib.ctg_push(session, C.byref(Buffer(C.sizeof(Buffer), None, 0))))
        self.assertEqual(-7, self.lib.ctg_flush(session))

    def test_borrowed_input_and_results_outlive_session_and_cartridge(self):
        cartridge = self.open()
        session = self.session(cartridge)
        self.assertEqual(0, self.push(session, '{"reply":"hé"}'.encode()))
        self.assertEqual(0, self.lib.ctg_flush(session))
        held = []
        while True:
            code, result = self.pull(session)
            self.assertEqual(0, code)
            if not result.value: break
            held.append((result, self.payload(result)))
        self.assertEqual("hé", json.loads(held[-1][1])["text"])
        self.assertEqual(0, self.end(session))
        self.assertEqual(0, self.close(cartridge))
        self.assert_reaped()
        for result, before in held:
            self.assertEqual(before, self.payload(result))
            self.lib.ctg_result_release(result)

    def test_one_active_stream_and_reuse_after_abandoning_session(self):
        cartridge = self.open()
        first, second = self.session(cartridge), self.session(cartridge)
        self.assertEqual(-7, self.close(cartridge))
        self.assertEqual(0, self.push(first, b'{"wait":true}'))
        self.assertEqual(0, self.lib.ctg_flush(first))
        self.assertEqual(0, self.push(second, b'{"reply":"second"}'))
        self.assertEqual(-7, self.lib.ctg_flush(second))
        self.assertEqual(0, self.end(first))
        self.assertEqual(0, self.lib.ctg_flush(second))
        self.assertEqual("second", self.drain(second)[-1][1]["text"])
        ids = [int(value) for value in (self.pack / "fake-request-ids.txt").read_text().splitlines()]
        self.assertEqual([1, 2], ids)

    def test_cross_thread_cancel_interrupts_blocked_pull_and_cartridge_reuses(self):
        cartridge = self.open()
        session = self.session(cartridge)
        self.assertEqual(0, self.push(session, b'{"wait":true}'))
        self.assertEqual(0, self.lib.ctg_flush(session))
        code, ready = self.pull(session)
        self.assertEqual(0, code)
        self.assertEqual("ready", json.loads(self.payload(ready))["type"])
        self.lib.ctg_result_release(ready)
        answer = []
        thread = threading.Thread(target=lambda: answer.append(self.pull(session)))
        thread.start()
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())
        self.assertEqual(0, self.lib.ctg_cancel(session))
        thread.join(3)
        self.assertFalse(thread.is_alive(), "cancel deadlocked behind pull")
        self.assertEqual(-6, answer[0][0])
        self.assertFalse(answer[0][1].value)
        self.assertEqual(-6, self.lib.ctg_flush(session))
        self.assertEqual(0, self.end(session))
        next_session = self.session(cartridge)
        self.assertEqual(0, self.push(next_session, b'{}'))
        self.assertEqual(0, self.lib.ctg_flush(next_session))
        self.assertEqual("final", self.drain(next_session)[-1][1]["type"])

    def test_cancel_before_run_and_invalid_json_leave_worker_reusable(self):
        cartridge = self.open()
        cancelled = self.session(cartridge)
        self.assertEqual(0, self.lib.ctg_cancel(cancelled))
        self.assertEqual(-6, self.push(cancelled, b'{}'))
        self.assertEqual(-6, self.pull(cancelled)[0])
        self.assertEqual(0, self.end(cancelled))
        invalid = self.session(cartridge)
        self.assertEqual(0, self.push(invalid, b'not-json'))
        self.assertEqual(0, self.lib.ctg_flush(invalid))
        self.assertEqual(-1, self.pull(invalid)[0])
        self.assertEqual(-7, self.pull(invalid)[0])
        self.assertEqual(0, self.end(invalid))
        valid = self.session(cartridge)
        self.assertEqual(0, self.push(valid, b'{}'))
        self.assertEqual(0, self.lib.ctg_flush(valid))
        self.drain(valid)

    def test_worker_protocol_errors_and_broken_pipe_are_contained(self):
        for mode in ("wrong_id", "truncated", "early_done", "exit_after_loaded"):
            with self.subTest(mode=mode):
                self.mode(mode)
                cartridge = self.open()
                session = self.session(cartridge)
                self.assertEqual(0, self.push(session, b'{}'))
                if mode == "exit_after_loaded": time.sleep(0.05)
                flushed = self.lib.ctg_flush(session)
                if flushed == 0:
                    self.assertEqual(-5, self.pull(session)[0])
                else:
                    self.assertEqual(-5, flushed)
                self.assertEqual(0, self.end(session))
                self.assertEqual(0, self.close(cartridge))
                self.assert_reaped()

    def test_startup_failures_and_bounded_teardown_reap_owned_child(self):
        for mode in ("startup_error", "startup_oversized"):
            self.mode(mode)
            cartridge = C.c_void_p()
            self.assertEqual(-4, self.lib.ctg_open(os.fsencode(self.pack), None, C.byref(cartridge)))
            self.assertFalse(cartridge.value)
            self.assert_reaped()
        self.mode("ignore_stop")
        cartridge = self.open()
        started = time.monotonic()
        self.assertEqual(0, self.close(cartridge))
        self.assertLess(time.monotonic() - started, 5)
        self.assert_reaped()

    def test_persistent_nested_worker_cleanup_on_startup_failure_and_forced_close(self):
        for mode in ("nested_startup_error", "nested_ignore_stop", "nested_graceful_stop"):
            with self.subTest(mode=mode):
                self.mode(mode)
                marker = self.pack / "fake-nested-worker.pid"
                marker.unlink(missing_ok=True)
                started = time.monotonic()
                if mode == "nested_startup_error":
                    cartridge = C.c_void_p()
                    self.assertEqual(-4, self.lib.ctg_open(os.fsencode(self.pack), None, C.byref(cartridge)))
                    self.assertFalse(cartridge.value)
                else:
                    cartridge = self.open()
                    self.assertEqual(0, self.close(cartridge))
                self.assertLess(time.monotonic() - started, 6)
                self.assert_reaped()
                pid = int(marker.read_text())
                # A non-child descendant is reaped by its OS parent; a zombie
                # has already exited and cannot retain a GPU context.
                stopped = False
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try: os.kill(pid, 0)
                    except ProcessLookupError:
                        stopped = True
                        break
                    stat = Path(f"/proc/{pid}/stat")
                    try:
                        if stat.read_text().rpartition(") ")[2].split()[0] == "Z":
                            stopped = True
                            break
                    except FileNotFoundError: pass
                    time.sleep(0.01)
                self.assertTrue(stopped, f"Nested worker {pid} is still running")
                if mode == "nested_graceful_stop":
                    self.assertEqual("nested child stopped before JVM exit", (self.pack / "fake-graceful-stop.txt").read_text())

    def test_worker_launch_when_host_standard_descriptors_are_closed(self):
        code = '''import ctypes as C, os, sys
library, pack = sys.argv[1:]
lib = C.CDLL(library)
lib.ctg_open.argtypes = [C.c_char_p, C.c_void_p, C.POINTER(C.c_void_p)]
lib.ctg_close.argtypes = [C.c_void_p]
for fd in (0, 1, 2):
    try: os.close(fd)
    except OSError: pass
handle = C.c_void_p()
assert lib.ctg_open(os.fsencode(pack), None, C.byref(handle)) == 0
assert handle.value
assert lib.ctg_close(handle) == 0
'''
        result = subprocess.run([sys.executable, "-c", code, str(self.library), str(self.pack)],
                                capture_output=True, timeout=10)
        self.assertEqual(0, result.returncode)
        self.assert_reaped()


if __name__ == "__main__":
    unittest.main()
