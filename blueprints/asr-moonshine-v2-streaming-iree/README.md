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
| Targets | `vulkan-armv7`, `cpu-armv7` (Android `armeabi-v7a`), `vulkan-arm64`, `cpu-arm64` (Android `arm64-v8a`); `cpu-sl26xx` (Linux aarch64, Cortex-A55, compiled by the Torq-fork compiler `skainet/iree-compiler-torq` because the SL26xx board runtime rejects upstream-IREE modules); `torq-sl26xx` (the same, with the encoder on the Torq NPU) |
| Toolchain | SKaiNET 0.61.0, SKaiNET-transformers 0.62.2 (exporter), 0.63.1 (runtime: incremental frontend), IREE tools 3.11.0 |

## What you will build

A signed cartridge `pack_dir` holding, per selected language, five compiled graphs (`frontend`, `encoder`,
`adapter`, `prefill`, `step`), one parameter archive with the decoder weights (not on targets whose decoder graphs
are int8, see Step 5), the decoder embedding table and the detokenizer table, plus the streaming runtime library for the target ABI — and a descriptor that states what was
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

Numerics: fp32 throughout; nothing is quantized at export. The export fails if a tensor of the checkpoint was not used.
A target can ask for int8 weights afterwards (Step 5).

## Step 4 — The parameter archive (`convert`: IREE tools)

```
./gradlew :blueprints:asr-moonshine-v2-streaming-iree:parametersDe -Pprofile=profiles/mine.json
```

`iree-compile --iree-hal-target-backends=llvm-cpu --iree-opt-export-parameters=model=params.irpa prefill.mlir` lifts the
decoder weights out of the graph into one archive. The archive is independent of the target and identical whether
taken from the `prefill` or the `with_past` graph, so it is produced once per language and shared. (The
throw-away host `.vmfb` of this step is discarded.) The step's product is the language's weights directory
(`weights<Lang>`): `dec_embed.bin`, plus `params.irpa` unless both decoder graphs are int8 (Step 5).

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

Three `convert` rewrites run before the compile when the target asks for them:

- `params.conv1d_im2col = "true"`: the frontend's two strided 1-D convolutions as im2col + one matmul each, same
  arithmetic (`im2colFrontend<Lang>`).
- `params.int8_weights = "encoder,adapter,prefill,with_past"` (the graphs to rewrite): every rank-2 f32 constant with at
  least 16384 elements becomes int8 with one symmetric scale per row, dequantized in the graph (`int8<Lang><Graph>`).
  Those graphs compile with `--iree-opt-const-expr-hoisting=false --iree-opt-const-eval=false`, so the weights stay
  int8 inside the `.vmfb`. When both decoder graphs are listed they read no parameter archive, so the archive is
  neither compiled against nor packed (86 MB less for the English flavor).
- `params.int8_activations = "with_past"` (a subset of `int8_weights`): those graphs' weight matmuls also quantize their
  activation per row (`max|row| / 127`) and multiply int8 by int8 into i32, both scales applied to the result. The CPU
  backend otherwise dequantizes the weights inside every matmul on every call; this way the matmuls use the
  dot-product ukernel. They compile with `--iree-global-opt-data-tiling --iree-llvmcpu-enable-ukernels=all` instead,
  which packs the int8 weights at compile time. The activation rounding changes the numerics a little: check the
  transcripts per graph.

- `params.npu_graphs = "frontend,encoder"` (the graphs to run on the Torq NPU, not also in `int8_weights`): the whole graph in
  bf16, shaped around what Torq 2.3.0 gets right (`torqBf16<Lang><Graph>`): weight matmuls on pre-transposed bf16
  constants (an im2col conv weight as its folded `[out, k·in]` matrix), sum reductions as matmuls with an all-ones
  matrix, optimization barriers after every matmul, transpose and strided slice, and after every elementwise op that
  feeds another one or a transpose (Torq runs an elementwise dispatch on the NPU only when it holds a single op).
  Those graphs compile with the image's `compile-torq --target <params.torq_hw>` and
  `--torq-disable-css --iree-scheduling-optimize-bindings=false --iree-flow-inline-constants-max-byte-length=4194304`.

- `params.cross_kv_cache = "true"` (for a runtime with a cross-K/V cache; `crossKv<Lang>`): prefill computes every
  layer's cross-attention K and V from all 256 padded memory rows on each restart, though a row's K/V never change
  once it is final. The step takes those projections out of prefill, which then reads the cached K/V instead of the
  memory, and adds them to the encoder module as a second function, `moonshine_v2_cross_kv`: one matmul from a
  window of adapter rows (`[1, 64, 320]`) to every layer's K and V side by side (`[1, 64, 12·320]`; output j in
  columns `j·320…`, head h at `h·40`). The runtime calls it after the adapter and splits the heads as it copies the
  finalized rows into its cache. It runs on the encoder's device; the adapter and the decoder-step graph are
  unchanged.

`cpu-sl26xx` sets the first two. On its Cortex-A55 the int8 weights take the streaming real-time factor from 1.48 to
0.95, with identical finals on 16 test clips. `torq-sl26xx` moves the frontend and the encoder to the NPU
(`npu_graphs = "frontend,encoder"`, int8 for the decoder graphs and the adapter): on an SL2610 with Torq 2.3.0 they
match their f32 graphs at cosine 0.99996 and 0.9995, finals are identical on 12 test clips, and the real-time factor
goes from 0.77 (`cpu-sl26xx`) to 0.57; the frontend takes 48 ms per window instead of 185 ms. The pack's descriptor
has to place both graphs on the `torq` device. With `cross_kv_cache` as well, prefill takes 32 ms per call instead of
166 ms and the K/V of a window 4.5 ms on the NPU; the real-time factor goes to 0.44 and the final arrives ~450 ms
after the last sample (was ~580 ms). `int8_activations = "with_past"` then takes a decoder step from 29 to 17 ms
(finals identical on the 12 test clips): real-time factor 0.35, final ~360 ms after the last sample. With prefill in
the list as well, 2 of the 12 finals changed, so it stays weight-only.

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
