#!/usr/bin/env python3
"""Independent ctypes consumer for the ABI evaluation candidate (not ctg-conform).

Loads only the C library. Message data crosses ctg_push; template rendering,
tokenization and generation remain behind the ABI. No reference fixtures are
available to this execution script. The saved report is validated separately.
"""
from __future__ import annotations

import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time


class Options(C.Structure):
    _fields_ = [("size", C.c_size_t), ("flags", C.c_uint32)]


class Buffer(C.Structure):
    _fields_ = [("size", C.c_size_t), ("data", C.c_void_p), ("len", C.c_size_t)]


class Preflight(C.Structure):
    _fields_ = [("size", C.c_size_t), ("ok", C.c_int32), ("detail", C.c_char_p)]


def identity(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return {"file": str(p), "sha256": h.hexdigest(), "size_bytes": p.stat().st_size}


def json_object(raw):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    assert isinstance(parsed, dict), "Expected a JSON object"
    return parsed


def canonical(value, ensure_ascii=True):
    # Matches the pinned manifest verifier's canonical descriptor digest.
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=ensure_ascii, allow_nan=False).encode("utf-8")


def check_package_identity(pack, library):
    """Check the two load-identity artifacts; this is not signature/inventory validation."""
    pack, library = pack.resolve(), library.resolve()
    raw_descriptor = (pack / "descriptor.json").read_bytes()
    raw_manifest = (pack / "manifest.json").read_bytes()
    descriptor, manifest = json_object(raw_descriptor), json_object(raw_manifest)
    canonical_descriptor = canonical(descriptor)
    assert manifest.get("kind") == "cartridge" and manifest.get("digest_alg") == "sha256", "Unexpected manifest kind/digest"
    assert manifest.get("id") == descriptor.get("id") and manifest.get("version") == descriptor.get("version"), "Manifest descriptor identity differs"
    assert manifest.get("descriptor_digest") == "sha256:" + hashlib.sha256(canonical_descriptor).hexdigest(), "Manifest canonical descriptor digest differs"
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list) and all(isinstance(a, dict) for a in artifacts), "Missing artifact inventory"
    paths = [a.get("path") for a in artifacts]
    assert all(isinstance(path, str) for path in paths) and len(paths) == len(set(paths)), "Duplicate or invalid artifact paths"
    descriptors = [a for a in artifacts if a.get("role") == "descriptor"]
    runtimes = [a for a in artifacts if a.get("role") == "runtime"]
    assert len(descriptors) == len(runtimes) == 1, "Expected one descriptor and one native runtime artifact"
    observed_descriptor = {"file": str(pack / "descriptor.json"), "sha256": hashlib.sha256(raw_descriptor).hexdigest(),
                           "size_bytes": len(raw_descriptor)}
    observed_library = identity(library)
    for entry, path, observed in ((descriptors[0], "descriptor.json", observed_descriptor),
                                  (runtimes[0], library.relative_to(pack).as_posix(), observed_library)):
        assert entry.get("path") == path, "Manifest artifact path differs from loaded artifact"
        assert entry.get("digest") == "sha256:" + observed["sha256"], "Manifest raw artifact digest differs"
        assert entry.get("size") == observed["size_bytes"], "Manifest artifact size differs"
    return canonical_descriptor, {
        "descriptor": observed_descriptor, "library": observed_library,
        "manifest": {"file": str(pack / "manifest.json"), "sha256": hashlib.sha256(raw_manifest).hexdigest(),
                     "size_bytes": len(raw_manifest)},
        "package_identity": {"descriptor_canonical_sha256": hashlib.sha256(canonical_descriptor).hexdigest(),
                             "descriptor_raw_artifact_checked": True, "runtime_raw_artifact_checked": True,
                             "manifest_descriptor_identity_checked": True,
                             "signature_verification": False, "full_artifact_inventory_checked": False},
    }


def check_embedded_descriptor(embedded, expected_canonical):
    assert isinstance(embedded, bytes), "Missing embedded descriptor"
    parsed = json_object(embedded)
    assert canonical(parsed) == expected_canonical, "Embedded descriptor JSON differs"
    assert embedded in (canonical(parsed), canonical(parsed, ensure_ascii=False)), "Embedded descriptor is not canonical JSON"
    return {"embedded_descriptor_canonical_equivalent": True,
            "embedded_descriptor_sha256": hashlib.sha256(embedded).hexdigest()}


def bind(library):
    lib = C.CDLL(str(library))
    signatures = {
        "ctg_abi_version": (C.c_uint32, []),
        "ctg_descriptor_json": (C.c_char_p, []),
        "ctg_preflight": (C.c_int, [C.c_char_p, C.POINTER(Preflight)]),
        "ctg_open": (C.c_int, [C.c_char_p, C.POINTER(Options), C.POINTER(C.c_void_p)]),
        "ctg_close": (C.c_int, [C.c_void_p]),
        "ctg_session_begin": (C.c_int, [C.c_void_p, C.POINTER(Options), C.POINTER(C.c_void_p)]),
        "ctg_session_end": (C.c_int, [C.c_void_p]),
        "ctg_push": (C.c_int, [C.c_void_p, C.POINTER(Buffer)]),
        "ctg_pull": (C.c_int, [C.c_void_p, C.POINTER(C.c_void_p)]),
        "ctg_flush": (C.c_int, [C.c_void_p]),
        "ctg_cancel": (C.c_int, [C.c_void_p]),
        "ctg_infer": (C.c_int, [C.c_void_p, C.POINTER(Buffer), C.POINTER(C.c_void_p)]),
        "ctg_result_get_kind": (C.c_int, [C.c_void_p]),
        "ctg_result_get_payload": (C.c_int, [C.c_void_p, C.POINTER(Buffer)]),
        "ctg_result_release": (None, [C.c_void_p]),
        "ctg_last_error": (C.c_char_p, [C.c_void_p]),
    }
    for name, (result, arguments) in signatures.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = result, arguments
    return lib


def execute(pack, library, request_dir, observed=None):
    pack, library = pack.resolve(), library.resolve()
    report = {
        "status": "running", "scope": "native_runtime_abi_independent_consumer",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_environment": {"system": platform.system(), "machine": platform.machine(),
                                  "kernel": platform.release()},
        "library": identity(library), "descriptor": identity(pack / "descriptor.json"),
        "worker_configuration": identity(pack / "artifacts/runtime/worker.json"),
        "consumer": identity(__file__), "cases": [], "negative_cases": [],
        "performance_benchmark": False, "task_quality_qualified": False,
        "sealed_cartridge": False, "full_spec_conformance": False,
    }
    if observed is not None:
        observed.update(report)
        report = observed
    expected_descriptor, identities = check_package_identity(pack, library)
    report.update(identities)
    lib = bind(library)
    assert lib.ctg_abi_version() == 1, "ABI version differs from pinned 0.1"
    report["package_identity"].update(check_embedded_descriptor(lib.ctg_descriptor_json(), expected_descriptor))
    pf = Preflight(C.sizeof(Preflight), 0, None)
    assert lib.ctg_preflight(os.fsencode(pack), C.byref(pf)) == 0 and pf.ok == 1, (pf.ok, pf.detail)
    report["preflight"] = {"status": 0, "ok": pf.ok, "detail": pf.detail.decode() if pf.detail else None}
    cartridge = C.c_void_p()
    started = time.monotonic()
    status = lib.ctg_open(os.fsencode(pack), None, C.byref(cartridge))
    assert status == 0 and cartridge.value, (status, lib.ctg_last_error(None))
    report["open_duration_seconds"] = time.monotonic() - started
    retained = []
    cleanup_safe = True

    def check(status, handle, expected=0):
        assert status == expected, (status, expected, lib.ctg_last_error(handle))

    def begin():
        session = C.c_void_p()
        check(lib.ctg_session_begin(cartridge, None, C.byref(session)), cartridge)
        assert session.value
        return session

    def push(session, payload):
        memory = C.create_string_buffer(payload)
        view = Buffer(C.sizeof(Buffer), C.cast(memory, C.c_void_p), len(payload))
        status = lib.ctg_push(session, C.byref(view))
        C.memset(memory, 0x58, len(payload))  # The ABI may not retain caller storage.
        return status

    def payload(result):
        view = Buffer(C.sizeof(Buffer), None, 0)
        check(lib.ctg_result_get_payload(result, C.byref(view)), None)
        return C.string_at(view.data, view.len)

    def pull(session, keep=False):
        result = C.c_void_p()
        status = lib.ctg_pull(session, C.byref(result))
        if status != 0:
            assert not result.value, "Failure returned an owned result"
            return status, None
        if not result.value:
            return status, None
        raw = payload(result)
        event = {"kind": lib.ctg_result_get_kind(result), "payload": json.loads(raw)}
        if keep:
            retained.append((result, raw))
        else:
            lib.ctg_result_release(result)
        return status, event

    def run(label, request, original=None):
        session = begin()
        raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
        events = []
        start = time.monotonic()
        try:
            status, empty = pull(session)
            check(status, session); assert empty is None
            for chunk in (raw[:1], raw[1:7], raw[7:]):
                check(push(session, chunk), session)
            check(lib.ctg_flush(session), session)
            check(push(session, b"x"), session, -7)
            prior_error = lib.ctg_last_error(session)
            for index in range(1024):
                status, event = pull(session, keep=(not retained))
                check(status, session)
                if event is None:
                    break
                events.append(event)
            else:
                raise AssertionError("Stream did not drain")
            # Successful calls must not replace a saved diagnostic.
            assert lib.ctg_last_error(session) == prior_error
            check(lib.ctg_flush(session), session, -7)
            status, empty = pull(session)
            check(status, session); assert empty is None
        finally:
            check(lib.ctg_session_end(session), None)
        record = {"id": label, "request": request, "request_sha256": hashlib.sha256(raw).hexdigest(),
                  "events": events, "duration_seconds": time.monotonic() - start,
                  "borrowed_input_overwritten_after_each_push": True, "drained_twice": True}
        if original:
            record["original_request"] = identity(original)
        report["cases"].append(record)
        return record

    try:
        for label in ("arithmetic", "system", "multiturn"):
            path = request_dir / f"{label}.json"
            source = json.loads(path.read_bytes())
            assert source["decoder_policy"] == "HUGGING_FACE_5_COMPAT"
            run(label, {"messages": source["messages"], "max_output_tokens": source["max_new_tokens"]}, path)

        for label, raw in (
            ("malformed-json", b"{"),
            ("invalid-utf8", b"\xff"),
            ("model-knob", b'{"messages":[{"role":"user","content":"hi"}],"decoder_policy":"CHECKPOINT"}'),
            ("unsupported-role", b'{"messages":[{"role":"tool","content":"hi"}]}'),
            ("unsupported-sampling", b'{"messages":[{"role":"user","content":"hi"}],"temperature":0.9}'),
            ("duplicate-key", b'{"messages":[],"messages":[{"role":"user","content":"hi"}]}'),
        ):
            session = begin()
            try:
                check(push(session, raw), session)
                check(lib.ctg_flush(session), session)
                status, event = pull(session)
                check(status, session, -1); assert event is None
                error = lib.ctg_last_error(session)
                assert error
                report["negative_cases"].append({"id": label, "status": status, "detail": error.decode()})
            finally:
                check(lib.ctg_session_end(session), None)

        session = begin()
        try:
            check(lib.ctg_cancel(session), session)
            for call in (lambda: push(session, b"{}"), lambda: lib.ctg_flush(session), lambda: pull(session)[0]):
                check(call(), session, -6)
            report["cancel_before_flush"] = "passed"
        finally:
            check(lib.ctg_session_end(session), None)

        # Actual model work, not a fake worker: cancel while the owner calls pull.
        session = begin()
        cancel_observation = {}
        thread = None
        try:
            request = dict(report["cases"][0]["request"], max_output_tokens=256)
            check(push(session, json.dumps(request).encode()), session)
            check(lib.ctg_flush(session), session)
            status, ready = pull(session)
            check(status, session)
            assert ready["payload"]["type"] == "ready"
            calls = []

            def cancel():
                time.sleep(0.01)
                cancel_observation["start"] = time.monotonic()
                cancel_observation["status"] = lib.ctg_cancel(session)
                cancel_observation["end"] = time.monotonic()

            thread = threading.Thread(target=cancel, name="abi-cancel", daemon=True)
            thread.start()
            for _ in range(1024):
                start = time.monotonic()
                status, event = pull(session)
                calls.append({"start": start, "end": time.monotonic(), "status": status})
                if status == -6:
                    break
                check(status, session)
                assert event is not None and event["payload"]["type"] != "final", "Completed before cancellation"
            else:
                raise AssertionError("Cancellation did not terminate the stream")
            thread.join(timeout=5)
            assert not thread.is_alive() and cancel_observation["status"] == 0
            assert any(c["start"] <= cancel_observation["start"] <= c["end"] and c["status"] == -6 for c in calls), \
                "Did not demonstrate cancellation of an in-flight call"
            report["cross_thread_cancel"] = {"status": "passed", "cancel_call": cancel_observation,
                                             "owner_pull_calls": calls, "terminal_status": -6}
        finally:
            if thread:
                thread.join(timeout=5)
                if thread.is_alive():
                    cleanup_safe = False
                    report["unsafe_cleanup_skipped"] = "Cancellation thread did not return; owned process watchdog must terminate this consumer."
                    raise RuntimeError("Refusing to free handles still used by cancellation thread")
            check(lib.ctg_session_end(session), None)

        run("after-cancel", report["cases"][0]["request"])
        report["one_cartridge_open_for_all_sessions"] = True
        report["fresh_session_after_cancel"] = "passed"
        for result, raw in retained:
            assert payload(result) == raw, "Result did not survive session end"
        report["result_survives_session_end"] = True
    finally:
        if cleanup_safe:
            closed = lib.ctg_close(cartridge)
            assert closed == 0, ("ctg_close failed", closed)
            for result, raw in retained:
                assert payload(result) == raw, "Result did not survive cartridge close"
                lib.ctg_result_release(result)
    report["result_survives_cartridge_close"] = True
    assert identity(library) == report["library"]
    assert identity(pack / "descriptor.json") == report["descriptor"]
    assert identity(pack / "manifest.json") == report["manifest"]
    report["status"] = "passed"
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output must be a new file")
    report = {}
    try:
        report = execute(args.pack, args.library, args.requests, observed=report)
    except Exception as error:
        report.update(status="failed", scope="native_runtime_abi_independent_consumer",
                      reason=repr(error), sealed_cartridge=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
