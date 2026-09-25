# Blueprint: `nlu-qwen-iree`

Function-calling NLU with **Qwen instruct models**, exported to StableHLO by SKaiNET-transformers under the
`qwen-kv-v1` KV-cache contract and compiled with IREE for a Vulkan GPU or the CPU. One transcript in, one function
call out. One blueprint, one **flavor** per checkpoint.

This directory is a [cartridge blueprint](https://skainet-developers.github.io/SKaiNET-cartridge/skainet-cartridge/blueprints.html):
the Kotlin API and the complete build recipe — **no weights, no compiled model, no native library, and no tool
catalog**. You materialize it into a cartridge with *your* profile.

| | |
|---|---|
| Recipe | [`blueprint.json`](blueprint.json) |
| Flavors | `qwen3-600m` — `Qwen/Qwen3-0.6B-GGUF` @ `23749fef…`, `Qwen3-0.6B-Q8_0.gguf` (639 446 688 bytes, `sha256:9465e63a…`) |
| Weights license | **Apache-2.0** (model card `license: apache-2.0`, [LICENSE at the pinned revision](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/23749fefcc72300e3a2ad315e1317431b06b590a/LICENSE)); redistribution allowed, no acceptance step |
| Code license | MIT |
| Kotlin API | Kotlin Multiplatform — `android`, `jvm`, `linuxX64`, `linuxArm64`, `macosArm64`. Contract, catalog, prompt and the resolve loop are common code; a **runtime binding exists for Android** |
| Targets | `vulkan-armv7` (IREE `vulkan-spirv`, `valhall4`), `cpu-armv7` (IREE `llvm-cpu`, arm32) — Android `armeabi-v7a` |
| Tool catalog | an **input** you supply ([schema](schema/tool-catalog.schema.json)); [`samples/toy-catalog.json`](samples/toy-catalog.json) is a three-function toy |
| Toolchain | SKaiNET 0.57.0, SKaiNET-transformers 0.57.0, IREE tools 3.11.0 |

## Status

| Check | Result |
|---|---|
| Recipe on the host (fetch, export, host-gather, parameters, compile, runtime) | runs end to end for `vulkan-armv7` and `cpu-armv7` |
| Compiled graphs vs. mainline llama.cpp (host IREE, `llvm-cpu`) | **32 of 32 greedy tokens identical** for `qwen3-600m` — SKaiNET-transformers `QwenVmfbParityTest`, evidence record `llm-inference/qwen/validation/L3-qwen3-06b-host-vmfb.json` |
| On a device | **not run yet.** There are no `reference_measurements`: nothing in this blueprint has been timed or scored on hardware |

A second flavor, `qwen25-500m` (Qwen2.5-0.5B-Instruct, template `qwen25`), is prepared in the code
(`QwenTemplate.QWEN25`) but not in the recipe: its attention projections carry biases, and SKaiNET's StableHLO
tracer does not yet externalize bias parameters, so the export is not correct. It becomes a flavor entry once that
is fixed upstream.

## What you will build

```
build/cartridge/nlu-qwen-iree-vulkan-armv7/
├── descriptor.json                       resolved from the template + your profile; flavors[] lists what is bundled
├── manifest.json                         digests, licenses, provenance.blueprint — signed by you
└── artifacts/
    ├── runtime/libskainet_iree_kv.so     KV-session JNI library (armeabi-v7a)
    ├── model/qwen3-600m/qwen-{with-past,prefill-with-past,prefill-at}.vmfb
    ├── weights/qwen3-600m/qwen-{with-past,prefill-with-past,prefill-at}.irpa    bf16, one archive per graph
    ├── other/qwen3-600m/manifest.json    qwen-kv-v1: layers, heads, key-value heads, RoPE base, mask heads
    ├── tokenizer/qwen3-600m/Qwen3-0.6B-Q8_0.gguf                            tokenizer source
    └── other/tool-catalog.json           your catalog
```

About 4.2 GB per flavor (three 1.2 GB archives, the 0.64 GB GGUF). It is side-loaded or delivered as an asset pack — never put into an AAR.

## Prerequisites

- JDK 21+, Git; Android SDK (`ANDROID_HOME`) with platform 36 — one of the API's targets is Android.
- Docker, for the *IREE tools (StableHLO → IREE)* steps (`skainet/iree-compiler:3.11.0`).
- ~12 GB free disk per flavor and 16 GB of heap for the export.

## Step 1 — Profile

Copy [`profiles/example.vulkan-armv7.qwen3-600m.json`](profiles/example.vulkan-armv7.qwen3-600m.json) to
`profiles/mine.json`, point `inputs.tool-catalog.path` at your catalog and state its license. `flavors` selects which
checkpoints the pack bundles. Leave `measurements` out until Step 7.

```
./gradlew :blueprints:nlu-qwen-iree:blueprintValidate -Pprofile=profiles/mine.json
```

## Step 2 — Download (`fetch`)

The plugin downloads `hf://Qwen/Qwen3-0.6B-GGUF` at the pinned revision into SKaiNET's data cache and verifies the
sha256 of the file. Apache-2.0 needs no acceptance record.

## Step 3 — Export to StableHLO (`export`)

```
./gradlew :blueprints:nlu-qwen-iree:exportQwen3600m -Pprofile=profiles/mine.json
```

One run of the released `QwenExportCli` (`sk.ainet.transformers:skainet-transformers-inference-qwen:0.57.0`) per
flavor, with `QWEN_GRAPH=all`. It derives the architecture from the GGUF (layer count, query and key-value heads,
head dimension, QK-norm, RoPE base, tied embeddings), traces the three graphs of the contract and writes
`qwen-<graph>.mlir`, `qwen-<graph>.safetensors` and the flavor's `manifest.json`.

| Graph | Exporter settings | Role at run time |
|---|---|---|
| `prefill-at` | `QWEN_SEQ=1024` | prefill the catalog prefix once; LM head on one selected position |
| `prefill-with-past` | `QWEN_CHUNK=32` | one 32-token utterance chunk against the cache (dynamic past length) |
| `with-past` | — | one decode token (dynamic past length) |

What the export does to the model, so the numbers can be trusted:

- **Weights**: the Q8_0 blocks are dequantized and stored as **bf16**; compute is **f32**. No quantization of its own.
- **Grouped-query attention** stays native: key/value caches are `[1, 8, past, 128]` for Qwen3-0.6B, never expanded
  to the 16 query heads.
- **Chunk mask** is head-shared, `[1, 1, 32, past + 32]`; SKaiNET 0.57.0 broadcasts it onto the grouped scores in
  the form IREE's Vulkan backend lowers.
- **argMax** runs over the logits padded from 151 936 to 153 600 entries with a large finite negative, because the
  Vulkan backend lowers the fused reduction only on multiples of 2048. The padded columns can never win; the real
  logits and the returned token are unchanged.

## Step 4 — Move the embedding lookup to the host (`convert`: host-gather)

Same rewrite as the FunctionGemma blueprint: each module's `stablehlo.gather` from the embedding table on `%arg0`
becomes an `%emb` argument that the runtime fills with rows read straight from the parameter archive. IREE's Vulkan
backend cannot compile the in-graph gather for this model; the numbers are unchanged by construction.

## Step 5 — Parameter archives (`convert`: IREE tools)

`iree-convert-parameters` turns each `qwen-<graph>.safetensors` into `qwen-<graph>.irpa`, scope `model`. Format change
only.

## Step 6 — Compile (`compile`: IREE tools)

```
./gradlew :blueprints:nlu-qwen-iree:compileQwen3600mWithPast \
          :blueprints:nlu-qwen-iree:compileQwen3600mPrefillWithPast \
          :blueprints:nlu-qwen-iree:compileQwen3600mPrefillAt -Pprofile=profiles/mine.json
```

`vulkan-armv7` compiles with `vulkan-spirv --iree-vulkan-target=valhall4`, `cpu-armv7` with `llvm-cpu` for arm32.

## Step 7 — Runtime, measure, pack, sign

```
export CARTRIDGE_SIGNING_KEY="$(cat ~/keys/my-dev-key.pem)"      # openssl genpkey -algorithm ed25519
./gradlew :blueprints:nlu-qwen-iree:materializeCartridge -Pprofile=profiles/mine.json
```

- **Runtime** — `libskainet_iree_kv.so` for the target ABI comes from the published
  `skainet-transformers-runtime-iree-android:0.57.0`, the first release whose session runs grouped-query models and
  builds the head-shared chunk mask.
- **Measure** — the first run stops with *"No `performance` for the descriptor"*. Push the pack to your device, run
  your utterances through `QwenNluCartridge`, and put what you measured into the profile. This blueprint has no
  reference measurements to compare against yet.
- **Memory** — the three bf16 archives are about 1.2 GB each for `qwen3-600m`. They are memory-mapped, but a 32-bit
  process has limited address space: check that all three map on your device before anything else.
- **Pack and sign** — as in the other blueprints: descriptor resolution, effective license, `pack_dir`, manifest with
  `provenance.blueprint`, Ed25519 signature.

## Step 8 — Use it from an app

```kotlin
val pack = PackDir(File(filesDir, "nlu-qwen").path)       // or QwenNluCartridge.packFromAssets(context, "nlu-qwen")
QwenNluCartridge(pack, cacheDir = cacheDir).use { nlu ->   // flavor = the pack's only one, or name it
    nlu.warmUp()                                            // once: prefill + snapshot of the catalog prefix
    when (val r = nlu.resolve("turn the lamp off", budgetMs = 15_000)) {
        is NluResolution.Call   -> println("${r.name} ${r.args}  (${r.timing.totalMs} ms)")
        is NluResolution.NoCall -> println("no call: ${r.text}")
        is NluResolution.Failed -> println("failed: ${r.reason}")
    }
}
```

## How the engine decodes

`QwenEngine` is the FunctionGemma loop with two Qwen-specific choices:

- **The assistant turn is forced open** with `<tool_call>\n{"name": "`. The model only produces the function name and
  its arguments: it cannot answer in prose first, and the call's fixed opening costs no decode steps.
- **Decoding stops on the text, at the brace that closes the call.** After a call, Qwen models often do not emit
  `</tool_call>` or `<|im_end|>` as their dedicated tokens and start a second, duplicate call instead, so a stop set
  of token ids lets them run on. The closing brace is always there; `CallScanner` finds it, ignoring braces inside
  JSON strings. `<|im_end|>` and `<|endoftext|>` still end a turn.

The call is parsed with SKaiNET-transformers' `ToolCallParser`, the name snapped to your catalog, the arguments
returned as strings. Qwen adds no BOS token. The chat template per flavor is the released
`QwenChatTemplate(enableThinking = false)` (Qwen3) or `Qwen25ChatTemplate` (Qwen2.5); `PromptSplitTest` checks
that prefix + utterance equals the one-shot rendering for both.

## Platforms

| Source set | Content |
|---|---|
| `commonMain` | `NluToolCallCartridge` / `NluResolution` · `ToolCatalog` · `QwenTemplate` / `QwenPrompt` · `CallScanner` · `PackDir` (flavor-aware) · **`QwenEngine`** against `KvBackend`, `NluTokenizer`, `PrefixIdCache` |
| `androidMain` | `QwenNluCartridge`: `KvBackend` over the released `IreeKvSession`, configured from the flavor's `manifest.json`; the GGUF tokenizer; `packFromAssets` |
| `jvm`, `linuxX64`, `linuxArm64`, `macosArm64` | compile and test the full API and engine. No runtime binding: the KV-session contract is implemented only by the Android JNI runtime |

To run on another platform, implement `KvBackend` and construct `QwenEngine` directly.
