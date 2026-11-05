# Native C ABI evaluation bridge

This source builds a Darwin or Linux shared library exporting the pinned Cartridge Runtime ABI 0.1. A caller uses `dlopen`, `ctg_open`, sessions and pull-based results. Internally, one private JVM child keeps the tested LFM2 model loaded. This is an evaluation bridge with an external Java requirement; it is not a claim of single-process native model execution, signed packaging, performance acceptance or complete cartridge conformance. ADR-002 rejects IPC as the public seam; this bridge retains the C ABI as its public seam and records the internal worker architecture explicitly.

The unchanged normative header and upstream MIT license are in `include/cartridge_abi.h` and `LICENSE.upstream`; `upstream.json` pins their source revision and hashes. The header wins over older prose where result lifetime differs: each returned result owns immutable payload bytes until `ctg_result_release`, including after session and cartridge close.

## Build

```sh
python3 native/build.py \
  --descriptor /path/to/descriptor.json \
  --output /new/path/liblfm2_cartridge.dylib
```

Use `.so` on Linux, `--cxx` to choose a C++17 compiler, and optional `--static-libstdcxx` on Linux. No CMake is required. The builder captures source/header bytes before compilation and compiles those copies, embeds the exact canonical descriptor bytes and creates the library plus `<library>.build.json` exclusively. The compiler needs libc, a C++ standard library and pthreads. Target architecture and dynamic symbol requirements must be checked separately before deployment; this script does not cross-compile automatically.

Descriptor encoding is sorted compact UTF-8 JSON without a trailing newline. When a materializer writes a formatted descriptor, also pass `--pack-descriptor /path/to/exact-materialized-descriptor.json`. The builder rejects duplicate keys and requires both inputs to canonicalize to the same JSON bytes. It embeds the canonical descriptor for `ctg_descriptor_json` and the exact materialized bytes for package identity checks, and records both captured input hashes. Omitting `--pack-descriptor` retains the original canonical-file behavior. This is a byte-identity check, not descriptor-schema or signed-manifest acceptance. Package preparation owns those separate gates. The pinned materializer's canonical digest uses ASCII escaping; descriptors in this evaluation use ASCII content, for which these encodings agree.

## Package and worker protocol

The library compares `pack_dir/descriptor.json` with the exact embedded pack descriptor bytes. It selects `pack_dir/artifacts/runtime/java` when that path exists, or requires `JAVA_HOME/bin/java` when it is absent. An invalid package path, including a dangling symlink, fails without falling back. There is no PATH search. Each preflight/open operation resolves the selected executable once to an absolute path; open uses that same resolved path for both its version probe and worker launch. A host using an external JRE supplies `JAVA_HOME` at deployment, so the public recipe contains no private machine path or symlink artifact. The host must keep that external installation stable for the operation; resolving a path does not pin all JRE files.

The selected Java executable launches using `posix_spawn` without a shell, passing Java preview/vector options, `-Xmx6g`, the `artifacts/jvm/lib/*` classpath and `sk.ainet.cartridge.generator.lfm2.worker.MainKt --pack-dir <absolute-pack>`. The worker reads `artifacts/runtime/worker.json`; checkpoint/template/decoder choices remain package-owned. Both preflight and open first run `java --enable-preview --add-modules jdk.incubator.vector -version`, requiring Java 21 and successful module resolution. They then launch the same packaged main with `--preflight --pack-dir <absolute-pack>` and a 256 MiB Java heap. The backend probe checks configuration/metadata and its declared external requirements without loading model weights into the execution engine. CPU and Vulkan readiness are reported as `eager-cpu` and `iree-vulkan`; the GPU worker owns its Python/IREE/device checks. The native layer requires exit status zero and one bounded JSON object with root-level `status: ready`, `model_loaded: false` and a supported backend. This does not claim to check undeclared memory requirements.

The Java version probe has a 15-second deadline and the packaged backend probe a 30-second deadline, each including child exit. Each permits at most 64 KiB of captured output. Backend stderr remains diagnostic output and cannot authorize readiness. Malformed, duplicated, nested-only or trailing readiness metadata fails. Each probe runs in a private process group. Failure paths close transport and send TERM then KILL to that owned group, including a nested Python device probe, before bounded leader reaping. The persistent JVM worker also runs in an owned process group, so forced cleanup covers its Python backend.

One socketpair carries frames with one byte of type, eight-byte unsigned request ID and four-byte payload length, all integers big-endian. Payloads are bounded to 4 MiB. RUN=1 carries one accumulated JSON request, CANCEL=2 an empty payload with that session's ID, and STOP=3 an empty payload with ID zero. Responses are LOADED=16 (ID zero), PARTIAL=17, FINAL=18, ERROR=19 (signed four-byte status plus UTF-8 detail), and empty DONE=20. Every request response carries its original monotonically allocated ID. Successful terminal output is FINAL then DONE; an error or cancellation is ERROR then DONE. The native layer never calls host callbacks.

## Lifecycle and limits

Sessions are cheap and independent, with one active generation per cartridge. `push` copies borrowed bytes into a bounded request; `flush` ends input and submits it. Before flush, `pull` returns no pending result. After flush, `pull` blocks for one worker frame, returning owned PARTIAL/FINAL results and then a drained null result after DONE. Push or repeated flush after submission fails. Wrong I/O-family calls return `CTG_E_UNSUPPORTED`.

Only `cancel` may run concurrently with the session's owner thread. It uses an independent command-write mutex and a request-specific cancel ID, so it can interrupt an owner blocked in pull without confusing later sessions. A cancelled session accepts only `session_end`. Session end cancels/drains an abandoned request before another session can use the model. Every call on a non-null handle presumes the caller still owns that valid handle; stale/random pointers are outside the ABI contract.

`session_end` consumes a valid session handle even when teardown fails; a failure diagnostic then resides in `ctg_last_error(NULL)`, so callers must not retry or inspect the freed handle. Cartridge close refuses while sessions exist. If an operating-system-level worker exit exceeds the bounded cleanup wait, close retains the cartridge handle and returns `CTG_E_BACKEND`; the caller may retry close. Results remain independently releasable in all cases. Successful calls do not clear prior per-handle diagnostic text.

Startup waits at most 120 seconds; ordinary response frames at most 180 seconds; observing cancellation shortens the entire frame's remaining wait to at most 30 seconds. Writes are bounded to five seconds, and session draining to 30 seconds. Normal close sends STOP and gives the JVM up to two seconds for graceful backend shutdown. Teardown then closes transport, polls exit without reaping the group leader, sends TERM then KILL to the owned group and uses bounded nonblocking leader reap checks. The leader remains unreaped until group signals finish, preventing process-group ID reuse during cleanup. Hosts must not separately reap library-owned children; descendants that have already exited are reaped by their OS parent. An OS process stuck after KILL cannot be guaranteed to exit within a deadline: failed startup/preflight children retain tracked PIDs for later nonblocking reap attempts. No detached reaper thread or host-global signal handler is installed. Socket-local SIGPIPE suppression protects the host from dead-worker writes.

## Focused protocol tests

```sh
python3 -m unittest discover -s native/tests -p 'test_bridge.py' -v
```

These compile the actual bridge and use a deliberately fake protocol worker. They cover ABI negotiation, canonical/formatted descriptor identity and input ambiguity, package Java precedence and external `JAVA_HOME` resolution, reuse of the probed executable, wrong Java version, borrowed input copying, result lifetime, stable errors, serialized session ownership, cancellation during blocked pull, reuse after cancellation/errors, malformed/truncated frames, broken pipes, bounded forced teardown and hosts with closed standard descriptors. They provide lifecycle evidence, not model inference or the unavailable upstream `ctg-conform` acceptance suite.
