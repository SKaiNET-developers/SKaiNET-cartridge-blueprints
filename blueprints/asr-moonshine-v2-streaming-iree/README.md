# Blueprint: `asr-moonshine-v2-streaming-iree`

Streaming speech-to-text: **Moonshine v2 tiny** (English, German), exported to StableHLO by SKaiNET-transformers
and compiled with IREE for a Vulkan GPU or the CPU. Audio in as it arrives, partial transcripts while you speak,
an exact final at the end of the utterance.

This directory is a [cartridge blueprint](https://skainet-developers.github.io/SKaiNET-cartridge/skainet-cartridge/blueprints.html):
the Kotlin API and the complete build recipe — **no weights, no compiled model, no native library**. You
materialize it into a cartridge with *your* profile.

| | |
|---|---|
| Recipe | [`blueprint.json`](blueprint.json) |
| Weights | `moonshine-ai/moonshine-streaming-tiny` @ `f8e9dfd8…` (English) and `moonshine-ai/moonshine-streaming-tiny-de` @ `928bfb92…` (German): `config.json`, `model.safetensors`, `tokenizer.json`, all pinned by SHA-256 |
| Weights license | **MIT** at the publisher for both checkpoints — verified on the model cards 2026-09-21. `redistribution: allowed`: nothing to accept, nothing gated. |
| Code license | MIT |
| Kotlin API | Kotlin Multiplatform — `android`, `jvm`, `linuxX64`, `linuxArm64`, `macosArm64`. Contract, pack locator and language selection are common code; a **runtime binding exists for Android** (see *Platforms*) |
| Flavors | `en`, `de` — one language per flavor; a profile selects one or both |
| Targets | `vulkan-armv7`, `cpu-armv7` (Android `armeabi-v7a`), `vulkan-arm64`, `cpu-arm64` (Android `arm64-v8a`) |
| Toolchain | SKaiNET 0.56.0, SKaiNET-transformers 0.56.1 (exporter) / 0.56.2 (runtime), IREE tools 3.11.0 |

## What you will build

A signed cartridge `pack_dir` holding, per selected language, five compiled graphs (`frontend`, `encoder`,
`adapter`, `prefill`, `step`), one parameter archive with the decoder weights, the decoder embedding table and the
detokenizer table, plus the streaming runtime library for the target ABI — and a descriptor that states what was
measured on *your* device.

## Prerequisites

- JDK 21+, Git; Android SDK (`ANDROID_HOME`) with platform 36 — one of the API's targets is Android.
- Docker, for the *IREE tools (StableHLO → IREE)* steps. Equivalent plain IREE commands are given below.
- ~4 GB free disk and 12 GB of heap for the export (the graphs carry the weights as constants while being traced).

## Step 1 — Profile

Copy [`profiles/example.vulkan-armv7.de.json`](profiles/example.vulkan-armv7.de.json) to `profiles/mine.json` and
pick your `target` and `flavors` (`["de"]`, `["en"]` or both). Leave `measurements` out until Step 6 —
materialization stops and tells you when it needs them.

```
./gradlew :blueprints:asr-moonshine-v2-streaming-iree:blueprintValidate -Pprofile=profiles/mine.json
```

## Step 2 — Download (`fetch`)

```
./gradlew :blueprints:asr-moonshine-v2-streaming-iree:blueprintFetch -Pprofile=profiles/mine.json
```

SKaiNET's data-source module fetches the three files of each selected flavor from `hf://` at the pinned revision
into the shared cache (`~/.cache/skainet/data`) and verifies every SHA-256 against `blueprint.json`; a changed
byte is fatal. Only selected flavors are fetched.

## Step 3 — Export to StableHLO (`export`)

```
./gradlew :blueprints:asr-moonshine-v2-streaming-iree:exportDe -Pprofile=profiles/mine.json     # or exportEn
```

The released `MoonshineV2ExportCli` (SKaiNET-transformers, Kotlin, no Python) reads the snapshot and writes the five
graphs of the streaming contract plus `dec_embed.bin` and `vocab.bin` to `build/blueprint/work/export/<lang>/`.
Geometry, per-layer attention bands and vocabulary size come from the checkpoint. The three shapes are the
runtime's compiled-in streaming geometry and are fixed in `blueprint.json`:

| | value | meaning |
|---|---|---|
| `MOONSHINE_FE_SAMPLES` | 21760 | frontend input: 68 frames = a 64-frame window + 4 frames of left context |
| `MOONSHINE_ENC_FRAMES` | 64 | encoder / adapter window, 1.28 s |
| `MOONSHINE_MAX_MEM` | 256 | decoder memory, padded and masked: 5.12 s of speech per utterance |

Numerics: fp32 throughout; nothing is quantized. The export fails if a tensor of the checkpoint was not used.

## Step 4 — The parameter archive (`convert`: IREE tools)

```
./gradlew :blueprints:asr-moonshine-v2-streaming-iree:parametersDe -Pprofile=profiles/mine.json
```

`iree-compile --iree-hal-target-backends=llvm-cpu --iree-opt-export-parameters=model=params.irpa prefill.mlir` lifts the
decoder weights out of the graph into one archive. The archive is independent of the target and identical whether
taken from the `prefill` or the `with_past` graph, so it is produced once per language and shared. (The
throw-away host `.vmfb` of this step is discarded.)

## Step 5 — Compile (`compile`: IREE tools, StableHLO → IREE)

```
./gradlew :blueprints:asr-moonshine-v2-streaming-iree:compileDeFrontend compileDeEncoder compileDeAdapter \
          compileDePrefill compileDeWithPast -Pprofile=profiles/mine.json
```

Per graph, for the profile's target. `frontend`, `encoder`, `adapter` embed their weights; `prefill` and `step`
(`with_past`) are compiled with `--iree-opt-export-parameters=model=…` so they reference the archive of Step 4.
Vulkan: `iree-compile --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target=valhall4`. CPU (arm32):
`--iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=armv7a-linux-androideabi29 --iree-llvmcpu-target-cpu=cortex-a55
--iree-llvmcpu-target-cpu-features=+neon --iree-llvmcpu-link-embedded=false --iree-llvmcpu-link-static=false
--iree-llvmcpu-stack-allocation-limit=1048576` (with a clang linker wrapper for the Android triple).

## Step 6 — Runtime, measure, pack, sign

```
./gradlew :blueprints:asr-moonshine-v2-streaming-iree:materializeCartridge -Pprofile=profiles/mine.json -PsigningKeyFile=<ed25519.pem>
```

`build-runtime` takes `libskainet_moonshine_stream.so` for the target ABI from the released SKaiNET-transformers
Android runtime — no C is compiled here. Then run the produced files on your device (the pack under
`build/cartridge/<id>/`), put what you measured into the profile:

```json
"measurements": { "performance": { "latency_p50_ms": 790, "rtf": 1.33, "measured_on": "<your device, your run>" } }
```

and run `materializeCartridge` again: it resolves the descriptor (flavors, languages, target, your numbers;
`quality` is `unmeasured` unless you supply a WER), packs `artifacts/<role>/<language>/…`, writes the manifest with
blueprint provenance and signs it. Verify with the spec's `verify_manifest.py`.

## Step 7 — Use it from an app

```kotlin
dependencies { implementation("sk.ainet.cartridge:asr-moonshine-v2-streaming-iree:0.1.0") }   // code only, no model
```

Android:

```kotlin
val pack = PackDir(File(filesDir, "asr-moonshine").path)      // or MoonshineAsrCartridge.packFromAssets(context, "asr-moonshine")
MoonshineAsrCartridge(pack, languageTag = "de-DE").use { asr ->  // picks the pack's flavor for the language
    val session = asr.open()                                     // loads the five graphs on the device (seconds)
    audio.chunks(80.milliseconds).forEach { pcm -> session.feed(pcm)?.let { partial -> show(partial) } }
    val text = session.finish()                                  // exact final; the session is ready for the next utterance
}
```

## Platforms

| Source set | Content |
|---|---|
| `commonMain` | `StreamingAsrCartridge` / `StreamingAsrSession` (the contract) · `PackDir` (kotlinx-io paths, per-language files) · `Language` (flavor selection with fallback) |
| `androidMain` | `MoonshineAsrCartridge`: the contract over SKaiNET-transformers' released `IreeMoonshineStream` (`libskainet_moonshine_stream.so`), `packFromAssets` |
| `jvm`, `linuxX64`, `linuxArm64`, `macosArm64` | compile and test the API. **No runtime binding yet**: the streaming runtime is an Android JNI library today; a Linux/macOS binding needs the same C runtime built for those hosts (an upstream item in SKaiNET-transformers). |

## Reproducibility

Checked on 2026-09-22 against a cartridge built by the previous, script-driven flow for the German flavor
(`vulkan-armv7`), from the same checkpoint revision:

| product | result |
|---|---|
| `params.irpa`, `dec_embed.bin`, `vocab.bin` | byte-identical |
| five `.vmfb` | same weights (`_const` blobs byte-identical), same dispatch set; each SPIR-V dispatch differs by a few bytes of compiler-flag-dependent metadata |
| on the device (arm32, Mali GPU, Vulkan, 80 ms chunks) | every partial and both finals identical, word for word, on a 1.24 s and a 4.0 s German clip; engine init 3.2 s, first partial 0.79 s, final +0.87 s |

The run is deterministic: repeating a step reproduces its bytes.

## Troubleshooting

- *`catalog prefix … does not fit`* — not this blueprint; that is the NLU one.
- *`nativeCreate failed` with `vulkan`* — the device has no Vulkan driver or the `.vmfb` was compiled for another
  GPU family; try the `cpu-*` target.
- *`pack … is missing artifacts/model/<lang>/…`* — the pack was materialized without that flavor; `PackDir.languages`
  says what it holds.
- *Partials look fine but the final is short* — a 4 s clip with repeated speech and no endpointing: the runtime's
  exact final re-decode is what the host should act on, and a VAD endpoint (a separate cartridge) cuts the audio where
  speech ends.
