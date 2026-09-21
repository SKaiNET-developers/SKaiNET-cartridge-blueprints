# Blueprint: `nlu-functiongemma-270m-iree`

Function-calling NLU on Android: **FunctionGemma 270M**, exported to StableHLO by SKaiNET-transformers under a
KV-cache contract and compiled with IREE for a Vulkan GPU or the CPU. One transcript in, one function call out.

This directory is a [cartridge blueprint](https://skainet-developers.github.io/SKaiNET-cartridge/skainet-cartridge/blueprints.html):
the Kotlin API and the complete build recipe — **no weights, no compiled model, no native library, and no tool
catalog**. You materialize it into a cartridge with *your* profile.

| | |
|---|---|
| Recipe | [`blueprint.json`](blueprint.json) |
| Weights | `unsloth/functiongemma-270m-it-GGUF` @ `1de0f7fc…`, `functiongemma-270m-it-Q8_0.gguf` (291 558 624 bytes, `sha256:8a17a4ed…`) — a Q8_0 GGUF of `google/functiongemma-270m-it` |
| Weights license | **Gemma Terms of Use** — <https://ai.google.dev/gemma/terms>. Not an OSI license. Verified on the model card 2026-09-21 (`license: gemma`, repository not gated; the base model is gated). The blueprint marks the source `acceptance-required`: nothing is downloaded until your profile records that you accepted the terms. |
| Code license | MIT |
| Targets | `vulkan-armv7` (IREE `vulkan-spirv`, `valhall4`), `cpu-armv7` (IREE `llvm-cpu`, arm32) — Android `armeabi-v7a` |
| Tool catalog | an **input** you supply ([schema](schema/tool-catalog.schema.json)); [`samples/toy-catalog.json`](samples/toy-catalog.json) is a three-function toy |
| Toolchain | SKaiNET 0.56.0, SKaiNET-transformers 0.56.0, IREE tools 3.11.0 |

## What you will build

```
build/cartridge/nlu-functiongemma-270m-iree-vulkan-armv7/
├── descriptor.json                       resolved from the template + your profile
├── manifest.json                         digests, licenses, provenance.blueprint — signed by you
└── artifacts/
    ├── runtime/libskainet_iree_kv.so     KV-session JNI library (armeabi-v7a)
    ├── model/gemma-{with-past,prefill-with-past,prefill-at}.vmfb
    ├── weights/gemma-{with-past,prefill-with-past,prefill-at}.irpa     3 × 536 MB, bf16
    ├── tokenizer/functiongemma-270m-it-Q8_0.gguf                       tokenizer source
    └── other/tool-catalog.json           your catalog
```

About 2.2 GB. It is side-loaded or delivered as an asset pack — never put into an AAR.

## Prerequisites

- JDK 21+, Git; Android SDK (`ANDROID_HOME`) with platform 36 — the API is an Android library.
- Docker, for the two *IREE tools (StableHLO → IREE)* steps. Equivalent plain IREE commands are given below.
- ~10 GB free disk and 8 GB of heap for the export.
- You have read the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the prohibited-use policy.

## Step 1 — Profile

Copy [`profiles/example.vulkan-armv7.json`](profiles/example.vulkan-armv7.json) to `profiles/mine.json` (git-ignored
names are up to you) and add the two things a repository can never contain:

```json
"license_acceptance": [
  { "source": "weights", "license": "Gemma Terms of Use", "terms_url": "https://ai.google.dev/gemma/terms",
    "accepted_by": "<your name or legal entity>", "accepted_at": "<date>" }
],
"measurements": { "performance": { "latency_p50_ms": 0, "measured_on": "<your device, your run>" } }
```

Leave `measurements` out until Step 7 — materialization stops and tells you when it needs them. Point
`inputs.tool-catalog.path` at your catalog (paths are relative to the profile) and state its license.

```
./gradlew :blueprints:nlu-functiongemma-270m-iree:blueprintValidate -Pprofile=profiles/mine.json
```

## Step 2 — Download (`fetch`)

```
./gradlew :blueprints:nlu-functiongemma-270m-iree:blueprintFetch -Pprofile=profiles/mine.json
```

Without the acceptance entry this stops *before any network access* and names the source, its revision, the
license and the terms URL. With it, the file is fetched by SKaiNET's `skainet-data-source` from
`hf://unsloth/functiongemma-270m-it-GGUF@1de0f7fc…`, verified against the pinned SHA-256 while streaming, and kept
in the shared cache `~/.cache/skainet/data` (one download per machine; `./gradlew --offline` afterwards). A stable
link appears at `build/blueprint/sources/weights/functiongemma-270m-it-Q8_0.gguf`. A `mirrors` entry in the profile
redirects the download — the digest stays the blueprint's.

## Step 3 — Export to StableHLO, and what happens to the numbers (`export`)

```
./gradlew :blueprints:nlu-functiongemma-270m-iree:exportWithPast \
          :blueprints:nlu-functiongemma-270m-iree:exportPrefillWithPast \
          :blueprints:nlu-functiongemma-270m-iree:exportPrefillAt -Pprofile=profiles/mine.json
```

Each task runs the released Kotlin exporter, `FunctionGemmaExportCli` from
`sk.ainet.transformers:skainet-transformers-inference-functiongemma:0.56.0`, resolved from Maven Central. It loads
the GGUF with SKaiNET's GGUF reader, traces the model in the SKaiNET DSL and writes, per graph,
`gemma-<graph>.mlir` (StableHLO) and `gemma-<graph>.safetensors` (the weights, *external* to the module).

| Graph | Exporter settings | Role at run time |
|---|---|---|
| `prefill-at` | `GEMMA_GRAPH=prefill_at GEN_SEQ=1024` | prefill the catalog prefix once; LM head on one selected position |
| `prefill-with-past` | `GEMMA_GRAPH=prefill_with_past GEMMA_CHUNK=32` | one 32-token utterance chunk against the cache |
| `with-past` | `GEMMA_GRAPH=with_past` | one decode token |

**Quantization, honestly.** The published file is Q8_0 (8-bit blocks). The exporter *dequantizes* it and stores the
weights as **bf16**; compute is **f32**. So this recipe performs no quantization of its own — it inherits the
publisher's Q8_0 rounding and then only casts. bf16 archives are the measured choice: f32 archives are twice the
size, were about 8× slower on the GPU and produced wrong tokens on the 32-bit CPU path. `GEMMA_QUANT=int8` exists
in the exporter and is *not* used here; an int8 variant would be a different blueprint version with its own
`quantize` step and its own measurements.

Open a `.mlir`: look for `stablehlo.` operations, and at the module header for the `skainet.schedule` attribute —
`parallel_dims = ["batch", "heads"]` per attention, structure only, never a core count. Apart from that header the
0.56.0 export of this model is line-for-line the module the original cartridge was built from.

## Step 4 — Move the embedding lookup to the host (`convert`: host-gather)

```
./gradlew :blueprints:nlu-functiongemma-270m-iree:hostGatherWithPast …PrefillWithPast …PrefillAt -Pprofile=…
```

An exported decoder begins with a gather of the 262 144 × 640 embedding table by token id. On a small GPU that
table is the largest buffer the graph touches, for a lookup the host can do by reading rows from the parameter
archive. The step rewrites three lines of each module:

```diff
-  func.func @gemma_with_past(%arg0: tensor<1xi32>, %arg1: …
+  func.func @gemma_with_past(%arg0: tensor<1xi32>, %emb: tensor<1x640xf32>, %arg1: …
-    %v237 = "stablehlo.gather"(%v0, %arg0) <{…}> : (tensor<262144x640xf32>, tensor<1xi32>) -> tensor<1x640xf32>
+    %ez = stablehlo.constant dense<0.0> : tensor<1x640xf32>
+    %v237 = stablehlo.add %emb, %ez : tensor<1x640xf32>
```

Numerics are unchanged: the graph receives exactly the rows the gather would have produced. The rewrite is Kotlin
(`HostGatherRewrite` in the plugin) and is tested byte-for-byte against the modules of the original build.

## Step 5 — Parameter archives (`convert`: IREE tools)

```
./gradlew :blueprints:nlu-functiongemma-270m-iree:parametersWithPast … -Pprofile=…
```

`safetensors → .irpa`, a format change only. Equivalent plain IREE command:

```
iree-convert-parameters --parameters=model=gemma-with-past.safetensors --output=gemma-with-past.irpa
```

## Step 6 — Compile (`compile`: IREE tools, StableHLO → IREE)

```
./gradlew :blueprints:nlu-functiongemma-270m-iree:compileWithPast … -Pprofile=…
```

The profile's target decides the backend (`blueprint.json` → `targets[].params`). Equivalent plain IREE commands:

```
# vulkan-armv7
iree-compile gemma-with-past.mlir --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target=valhall4 \
             -o gemma-with-past.vmfb
# cpu-armv7 — linked as a system .so, not an embedded ELF (32-bit ARM codegen needs bionic to resolve one symbol)
iree-compile gemma-with-past.mlir --iree-hal-target-backends=llvm-cpu \
             --iree-llvmcpu-link-embedded=false --iree-llvmcpu-link-static=false \
             --iree-llvmcpu-system-linker-path=<wrapper: clang --target=armv7a-linux-androideabi29 -fuse-ld=lld> \
             --iree-llvmcpu-target-triple=armv7a-linux-androideabi29 \
             --iree-llvmcpu-target-cpu=cortex-a55 --iree-llvmcpu-target-cpu-features=+neon \
             --iree-llvmcpu-stack-allocation-limit=1048576 \
             -o gemma-with-past.vmfb
```

(IREE 3.11.0. These are exactly the flags the containerized *IREE tools* pass; their image also provides the linker
wrapper. Without the image, `--iree-llvmcpu-system-linker-path` points at a small script that runs an NDK or LLVM
`clang` with that `--target` and `-fuse-ld=lld`.)

## Step 7 — Runtime, measure, pack, sign

```
export CARTRIDGE_SIGNING_KEY="$(cat ~/keys/my-dev-key.pem)"      # openssl genpkey -algorithm ed25519
./gradlew :blueprints:nlu-functiongemma-270m-iree:materializeCartridge -Pprofile=profiles/mine.json
```

- **Runtime** — `libskainet_iree_kv.so` for the target ABI is taken from the published
  `skainet-transformers-runtime-iree-android:0.56.0`; its source is `llm-runtime/iree-android/native` in
  SKaiNET-transformers.
- **Measure** — the first run stops with *"No `performance` for the descriptor"*. Push the pack to your device, run
  your utterances through `FunctionGemmaNluCartridge`, and put what you measured into the profile. The blueprint's
  `reference_measurements` say what its authors saw (about 5.4 s p50 per utterance on a 4-core arm32 device with a
  Mali GPU, 20 s on its CPU, ~60 s warm-up, ~1.5 GB RSS); they are never copied. Quality on *your* catalog is yours
  to measure — this small model is a runtime demonstrator, not an accuracy promise.
- **Pack and sign** — descriptor resolution, effective license (`MIT AND LicenseRef-Gemma-Terms-of-Use`, or
  `proprietary` with a proprietary catalog), `pack_dir`, manifest with `provenance.blueprint`, Ed25519 signature.
  Verify with the spec's `verify_manifest.py manifest.json my-dev-key=<public key>`.

## Step 8 — Use it from an app

```kotlin
dependencies { implementation("sk.ainet.cartridge:nlu-functiongemma-270m-iree:0.1.0") }   // code only
```

```kotlin
val pack = PackDir(File(filesDir, "nlu-functiongemma"))          // or PackDir.fromAssets(context, "nlu-functiongemma")
FunctionGemmaNluCartridge(pack, cacheDir = cacheDir).use { nlu ->
    nlu.warmUp()                                                  // once: prefill + snapshot of the catalog prefix
    when (val r = nlu.resolve("turn the lamp off", budgetMs = 8_000)) {
        is NluResolution.Call   -> println("${r.name} ${r.args}  (${r.timing.totalMs} ms)")
        is NluResolution.NoCall -> println("no call: ${r.text}")
        is NluResolution.Failed -> println("failed: ${r.reason}")
    }
}
```

The cartridge returns a function name and string arguments from *your* catalog. Turning that into an application
action is the host's job and never part of a cartridge.

## Reproducibility

Measured, not promised: these are the SHA-256 digests this recipe produced on the authors' machine (linux x86-64,
the pinned toolchain above). Six of the nine model artifacts and the runtime library were reproduced against an
independent build made weeks earlier, byte for byte; the three `prefill-at` artifacts were built once. If yours
differ, something in your toolchain does — which is exactly what the pins are for. The cartridge spec treats such digests as a *direction* (OQ11 / OQ-B2), so
they live here and not in `blueprint.json`.

| Artifact in the `pack_dir` | sha256 |
|---|---|
| `model/gemma-with-past.vmfb` — `vulkan-armv7` | `f6f3994817acdd62275115b0d05d2db47c9494283ccd8c0ff37ecd4847f305a5` |
| `model/gemma-with-past.vmfb` — `cpu-armv7` | `de7272e3e384669bced58043b856d0e42dd18dba81e0ad9d3c0f2c4f768576c5` |
| `weights/gemma-with-past.irpa` | `30eb5cc10ef24c1a6d4a765a2cba547d47f77babf122e53ec78ec0f2e16a9da4` |
| `model/gemma-prefill-with-past.vmfb` — `vulkan-armv7` | `9d37c737b39a222e2106d653bc70051145d73878edbabf8f8757c4138f1ccc74` |
| `model/gemma-prefill-with-past.vmfb` — `cpu-armv7` | `f731923a0e9484aae20b56d90667de85cf509ab279518e4168e417f384177148` |
| `weights/gemma-prefill-with-past.irpa` | `f8117650a6c4e3cfabc0548c94d7868e3ee7ece65051f2d9e167d11a285e725d` |
| `model/gemma-prefill-at.vmfb` — `vulkan-armv7` | `7fbcdb79e48d78df055b0ac176b277543058ac7993d61877c41b4eaab776cc54` |
| `model/gemma-prefill-at.vmfb` — `cpu-armv7` | `da92107ee20248b65a588de0ed1e63d2c9a86993407a5778449dc8e0e2a6a90d` |
| `weights/gemma-prefill-at.irpa` | `20a50766cda7f0bc3b3f2a1053cf4dcc77e911f1a0e297251f5b2da63be89cf6` |
| `runtime/libskainet_iree_kv.so` — `armeabi-v7a` | `3084e7b53055f7cba8061697e9820ee86c302bd0eaa4c8a60eab82a77b16842e` |

Where this recipe was checked against the hand-built cartridge it replaces: `with-past` and `prefill-with-past`
(both targets, modules and archives) and the runtime library are **byte-identical**. `prefill-at` differs on
purpose — since SKaiNET-transformers 0.56.0 the prefill path no longer expands the K/V heads (36 fewer
`stablehlo.concatenate`), with an unchanged function signature; it has to pass the on-device golden check before a
cartridge built from it replaces one built before 0.56.0.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `License acceptance required before anything is downloaded` | Step 1: add `license_acceptance` to your profile. |
| `Digest mismatch … do not proceed` | The publisher (or your mirror) serves different bytes at the pinned revision. |
| `IREE tools failed` | The image is not available: use the plain IREE commands above. |
| `catalog … does not fit the prefill graph (1024)` | The rendered catalog is too long; shorten descriptions or export `prefill-at` with a larger `GEN_SEQ`. |
| Wrong tokens on the CPU target | f32 archives on arm32 — keep the recipe's bf16. |
