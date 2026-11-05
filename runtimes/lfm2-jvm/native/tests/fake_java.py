#!/usr/bin/env python3
"""Protocol-only test worker. It never loads a model or starts a JVM."""
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import time

config = json.loads((Path(sys.argv[0]).parent / "worker.json").read_text())
mode = config.get("fake_mode", "normal")
if "-version" in sys.argv:
    if config.get("replace_java_link"):
        link = Path(config["replace_java_link"])
        link.unlink()
        link.symlink_to(config["replace_java_target"])
    print('openjdk version "17.0.1"' if mode == "wrong_java" else 'openjdk version "21.0.10"', file=sys.stderr)
    raise SystemExit(0)
pack = Path(sys.argv[sys.argv.index("--pack-dir") + 1])
if "--preflight" in sys.argv:
    (pack / "fake-preflight.pid").write_text(str(os.getpid()))
    if mode == "preflight_error":
        print("fake backend dependency unavailable", file=sys.stderr)
        raise SystemExit(1)
    if mode in ("preflight_hang", "preflight_oversized"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        nested = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_DFL); time.sleep(120)"])
        (pack / "fake-nested-preflight.pid").write_text(str(nested.pid))
        if mode == "preflight_oversized":
            sys.stdout.write("x" * 65537)
            sys.stdout.flush()
        while True:
            time.sleep(1)
    replies = {
        "preflight_invalid": "not JSON",
        "preflight_nested": '{"details":{"status":"ready","backend":"eager-cpu"}}',
        "preflight_duplicate": '{"status":"failed","status":"ready","backend":"eager-cpu"}',
        "preflight_trailing": '{"status":"ready","backend":"eager-cpu"} trailing',
        "preflight_unknown_backend": '{"status":"ready","backend":"unsupported"}',
        "preflight_model_loaded": '{"status":"ready","backend":"eager-cpu","model_loaded":true}',
        "preflight_model_missing": '{"status":"ready","backend":"eager-cpu"}',
    }
    print(replies.get(mode, json.dumps({"status": "ready", "backend": "iree-vulkan" if mode == "preflight_gpu" else "eager-cpu", "model_loaded": False,
                                       "details": {"no_model_loaded": True, "checked": [1, None, "metadata"]}})))
    raise SystemExit(0)
(pack / "fake-worker.pid").write_text(str(os.getpid()))
nested_worker = None
if mode.startswith("nested_"):
    ready = pack / "fake-nested-worker.pid"
    nested_worker = subprocess.Popen([sys.executable, "-c",
        "import os,signal,sys,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_DFL); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(120)", str(ready)])
    while not ready.exists(): time.sleep(0.001)
write_lock = threading.Lock()
active_lock = threading.Lock()
active = {}


def frame(kind, request_id, payload=b""):
    if isinstance(payload, dict):
        payload = json.dumps(payload, ensure_ascii=False).encode()
    return struct.pack(">BQI", kind, request_id, len(payload)) + payload


def send(*frames):
    with write_lock:
        sys.stdout.buffer.write(b"".join(frames))
        sys.stdout.buffer.flush()


def error(request_id, status, detail):
    send(frame(19, request_id, struct.pack(">i", status) + detail.encode()), frame(20, request_id))


def exact(length):
    data = bytearray()
    while len(data) < length:
        chunk = sys.stdin.buffer.read(length - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def generate(request_id, request, cancel):
    try:
        send(frame(17, request_id, {"type": "ready"}))
        if request.get("wait"):
            cancel.wait(10)
        if cancel.is_set():
            error(request_id, -6, "fake generation cancelled")
            return
        reply = request.get("reply", "hé")
        send(frame(17, request_id, {"type": "delta", "text": reply}),
             frame(18, request_id, {"type": "final", "text": reply, "finish_reason": "STOP", "tokens": [42, 7]}),
             frame(20, request_id))
    finally:
        with active_lock:
            active.pop(request_id, None)


if mode in ("startup_error", "nested_startup_error"):
    send(frame(19, 0, struct.pack(">i", -4) + b"fake load failure"))
    raise SystemExit(3)
if mode == "startup_oversized":
    sys.stdout.buffer.write(struct.pack(">BQI", 16, 0, 4 * 1024 * 1024 + 1))
    sys.stdout.buffer.flush()
    raise SystemExit(3)
send(frame(16, 0, {"fake": True}))
if mode == "exit_after_loaded":
    raise SystemExit(0)
if mode in ("ignore_stop", "nested_ignore_stop"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(1)
while True:
    header = exact(13)
    if header is None:
        break
    kind, request_id, length = struct.unpack(">BQI", header)
    payload = exact(length)
    if payload is None:
        break
    if kind == 3:
        if nested_worker:
            nested_worker.terminate()
            nested_worker.wait(timeout=2)
            (pack / "fake-graceful-stop.txt").write_text("nested child stopped before JVM exit")
        with active_lock:
            for item in active.values(): item.set()
        break
    if kind == 2:
        with active_lock:
            if request_id in active: active[request_id].set()
        continue
    if kind != 1:
        error(request_id, -1, "unknown command")
        continue
    with (pack / "fake-request-ids.txt").open("a") as stream:
        stream.write(str(request_id) + "\n")
    if mode == "wrong_id":
        send(frame(17, request_id + 1, {"type": "ready"}))
        continue
    if mode == "truncated":
        sys.stdout.buffer.write(struct.pack(">BQI", 17, request_id, 100) + b"short")
        sys.stdout.buffer.flush()
        raise SystemExit(3)
    if mode == "early_done":
        send(frame(20, request_id))
        continue
    try:
        request = json.loads(payload)
        if not isinstance(request, dict): raise ValueError("object required")
    except (ValueError, UnicodeError):
        error(request_id, -1, "fake invalid JSON")
        continue
    cancel = threading.Event()
    with active_lock:
        active[request_id] = cancel
    threading.Thread(target=generate, args=(request_id, request, cancel), daemon=True).start()
