# Blueprint: `asr-whisper-iree`

Speech-to-text for short utterances: **Whisper tiny** (English, German), exported to StableHLO by SKaiNET-transformers
and compiled with IREE for a Vulkan GPU or the CPU. One utterance of up to 4 s in, its transcript out.

This directory is a [cartridge blueprint](https://skainet-developers.github.io/SKaiNET-cartridge/skainet-cartridge/blueprints.html):
the Kotlin API and the complete build recipe — **no weights, no compiled model, no native library**. You
materialize it into a cartridge with *your* profile.

| | |
|---|---|
| Recipe | [`blueprint.json`](blueprint.json) |
| Weights | `openai/whisper-tiny.en` @ `87c71024…` (English) and `primeline/whisper-tiny-german-1224` @ `620446ea…` (German, a fine-tune of whisper-tiny): `config.json`, `model.safetensors`, `tokenizer.json`, all pinned by SHA-256 |
| Weights license | **Apache-2.0** at the publisher for both checkpoints — verified on the model cards 2026-10-21. `redistribution: allowed`: nothing to accept, nothing gated. |
| Code license | MIT |
| Kotlin API | Kotlin Multiplatform — `android`, `jvm`, `linuxX64`, `linuxArm64`, `macosArm64`: the batch contract, the pack locator (with the graph contract) and language selection. **No runtime binding yet** (see *Platforms*) |
| Flavors | `en`, `de` — one language per flavor; a profile selects one or both |
| Targets | `vulkan-armv7`, `cpu-armv7` (Android `armeabi-v7a`), `vulkan-arm64`, `cpu-arm64` (Android `arm64-v8a`) |
| Toolchain | SKaiNET 0.61.0, SKaiNET-transformers with `WhisperExportCli` (exporter), IREE tools 3.11.0 |

## What you will build

A signed cartridge `pack_dir` holding, per selected language, three compiled graphs (`encoder`, `prefill`, `step`),
**one** parameter archive with the decoder weights shared by `prefill` and `step`, the graph contract
(`whisper-contract.json`) and the detokenizer table (`tokenizer.json`) — and a descriptor that states what was
measured on *your* device.

## The model contract

| Graph | In | Out | Weights |
|---|---|---|---|
| `whisper_encoder` | log-mel `[1, 80, 400]` (4 s) | features `[1, 200, 384]` | inline |
| `whisper_prefill` | prompt ids `[n]` i32, features | logits `[1, n, vocab]`, per layer self K/V `[1, 48, 384]` (zero-padded) and cross K/V `[1, 200, 384]` | `params.irpa`, scope `model` |
| `whisper_step` | token, position, additive mask `[1, 1, 1, 48]`, write mask `[1, 48, 1]`, per layer self and cross K/V | logits `[1, 1, vocab]`, per layer updated self K/V | `params.irpa`, scope `model` |

The prompt is `<sot> <lang> <transcribe> <notimestamps>` (n = 4) for a multilingual checkpoint and `<sot>
<notimestamps>` (n = 2) for an English-only one; `whisper-contract.json` holds the exact ids, the end-of-text id,
the function names and the argument and result order. The KV cache is explicit graph I/O and the masks are computed
by the host — no boolean tensors in the graphs, which keeps them valid SPIR-V. Greedy decoding runs `prefill` once and
`step` per token until end-of-text or 48 positions.

**On the host:** the 80-bin log-mel spectrogram of the 16 kHz audio (Whisper's standard frontend: 25 ms window,
10 ms hop, log10 with Whisper's clamping and scaling), padded or cut to 400 frames; and detokenization with
`tokenizer.json`.

## Prerequisites

- JDK 21+, Git; Android SDK (`ANDROID_HOME`) with platform 36 — one of the API's targets is Android.
- Docker, for the *IREE tools (StableHLO → IREE)* steps.
- ~1 GB free disk and 8 GB of heap for the export.

## Step 1 — Profile

Copy [`profiles/example.vulkan-armv7.de.json`](profiles/example.vulkan-armv7.de.json) to `profiles/mine.json` and
pick your `target` and `flavors` (`["de"]`, `["en"]` or both). Leave `measurements` out until you have measured.

```
./gradlew :blueprints:asr-whisper-iree:blueprintValidate -Pprofile=profiles/mine.json
```

## Step 2 — Download (`fetch`)

```
./gradlew :blueprints:asr-whisper-iree:blueprintFetch -Pprofile=profiles/mine.json
```

SKaiNET's data-source module fetches the three files of each selected flavor from `hf://` at the pinned revision
and verifies every SHA-256 against `blueprint.json`; a changed byte is fatal.

## Step 3 — Export to StableHLO (`export`)

```
./gradlew :blueprints:asr-whisper-iree:exportDe -Pprofile=profiles/mine.json     # or exportEn
```

The released `WhisperExportCli` (SKaiNET-transformers, Kotlin, no Python) reads the snapshot and writes
`whisper_encoder.mlir`, `whisper_prefill.mlir`, `whisper_step.mlir`, `params.irpa` and `manifest.json` (packed as
`whisper-contract.json`) to `build/blueprint/work/export/<lang>/`. Geometry and vocabulary come from the
checkpoint; the two shapes are fixed in `blueprint.json`:

| | value | meaning |
|---|---|---|
| `WHISPER_AUDIO_CTX` | 200 | encoder positions after the stride-2 convolution: a 4 s window (the stock model uses 1500 = 30 s) |
| `WHISPER_MAX_POSITIONS` | 48 | decoder KV cache: prompt plus generated tokens |

All three graphs are traced in one session, so `prefill` and `step` reference their weights under the same keys and
share one archive (deduplicated, byte-identity asserted). Numerics: fp32; nothing is quantized.

## Step 4 — Compile (`compile`: IREE tools)

```
./gradlew :blueprints:asr-whisper-iree:compileDeEncoder :blueprints:asr-whisper-iree:compileDePrefill \
          :blueprints:asr-whisper-iree:compileDeStep -Pprofile=profiles/mine.json
```

`iree-compile` for the target's backend (`vulkan-spirv` with `--iree-vulkan-target`, or `llvm-cpu`). The prefill and
step modules keep their parameter references; a runtime loads `params.irpa` under scope `model`.

## Step 5 — Materialize (`pack`, `sign`)

```
./gradlew :blueprints:asr-whisper-iree:materializeCartridge -Pprofile=profiles/mine.json
```

## Platforms

The API is common code. There is **no released runtime binding** for the whisper contract yet: a host runs the
three modules with an IREE runtime of its own (`iree-run-module`, the IREE runtime C API, or its own JNI), in the
order and with the shapes `whisper-contract.json` names. The export's parity test in SKaiNET-transformers does exactly
that with `iree-run-module` and reproduces the reference greedy transcript from the compiled modules.

## Verification

The export is checked in SKaiNET-transformers against the German checkpoint pinned here: encoder cosine 0.99999 to
the reference pipeline, and the reference greedy tokens reproduced from the compiled modules.
