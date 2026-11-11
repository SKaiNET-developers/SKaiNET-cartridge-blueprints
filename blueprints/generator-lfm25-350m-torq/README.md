# generator-lfm25-350m-torq

Text generation with [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) on the Synaptics **Torq NPU**
(SL26xx): the model's single-token decode step as **one NPU dispatch**, the same graph for every position up to a
fixed capacity (64 tokens). The step is traced from the SKaiNET DSL with the projections as Q8_0 and the output head
as a Q4_0 copy of the tied embedding, kept packed as block-quant matmuls; Torq 2.3.0 lowers them to its mixed-quant
kernel. The embedding lookup, RoPE, the masks, sampling and stopping run on the host.

This module holds no model: the Kotlin API is the host side of the step contract, and the recipe builds the pack
from the publisher's checkpoint on your machine, together with its runtime: `artifacts/runtime/liblfm2_torq_cartridge.so`,
the Cartridge Runtime ABI over this API with the step on the NPU in-process
([`runtimes/lfm2-torq`](../../runtimes/lfm2-torq/README.md)), built from source. The weights are under the LFM Open License v1.0, which you accept
in your materialization profile (`license_acceptance`); the recipe never accepts it for you.

## Kotlin API (`commonMain`: JVM, Android, linuxX64, linuxArm64)

```kotlin
val ports = DecodeStepPorts.parse(packDir.resolve("artifacts/model/io.txt").readText())
val decoder = GreedyDecoder(
    ports,
    StepControls(headDim = ports.headDim, ropeTheta = 1e6, capacity = ports.capacity),
    EmbeddingRows(ports.vocabulary, ports.width, embedBf16Bytes),       // artifacts/weights/embed.bf16.bin
    runner,                                                             // your DecodeStepRunner over the Torq runtime
)
val result = decoder.generate(promptIds, maxNewTokens = 32, stopTokens = setOf(7))
```

- `DecodeStepPorts`: the argument and result order of the compiled function (`io.txt`), and which ports are state.
- `StepControls`: split-half RoPE, the additive attention mask `[1, C + 1]` (the filled prefix and the current
  token open, the rest `-1e30`) and the one-hot write mask `[1, C, 1]` for a position.
- `EmbeddingRows`: the bf16 embedding table, one row per token.
- `GreedyDecoder`: feeds the prompt one token per step, then its own arg-max, threading the conv histories and K/V
  from each call's results into the next call's arguments; stops at a stop token, the token budget or the capacity.
- `DecodeStepRunner`: one call of the compiled step, roles to values. It is the only device-specific part.

## The recipe

| Step | What it does |
|---|---|
| `fetch` | `hf://LiquidAI/LFM2.5-350M` at the pinned revision, every file SHA-256 verified |
| `export` (`exportDecodeStep`) | `Lfm2DecodeStepExportCli` from SKaiNET-transformers: projections loaded as Q8_0, a Q4_0 head, the fixed-capacity step traced with every packed matmul folded (93; one left unfolded fails the step); writes `decode_step.mlir`, `io.txt`, `embed.bf16.bin`, `export.json` |
| `torq-bf16-retype` (`torqBf16Retype`) | every f32 to bf16, arguments and results included: Torq's f32 host fallback returns NaN |
| `compile` (`compileDecodeStep`) | `compile-torq --input stablehlo-torq --target SL2610` in `skainet/iree-compiler-torq:2.3.0` |
| `build-torq-runtime` (`:runtimes:lfm2-torq:buildTorqNative`) | in `skainet/iree-runtime-torq:2.3.0`: the Torq-fork IREE runtime (built from source in the image) with its license set, the IREE session shim (`skainet_iree`) and the resolved descriptor's bytes (`resolveRuntimeDescriptor`) |
| `build-native` (`:runtimes:lfm2-torq:linkCartridgeReleaseSharedLinuxArm64`) | Kotlin/Native: `liblfm2_torq_cartridge.so`, the Cartridge Runtime ABI 0.1 for linux-aarch64 |
| `pack`, `sign` | the pack_dir with the runtime and its licenses, the module, port table, export manifest, embedding rows, tokenizer files and the model license |

```sh
./gradlew :blueprints:generator-lfm25-350m-torq:materializeCartridge -Pprofile=/private/path/torq-sl26xx.json
```

You need Docker with two local images, built from source and never pulled from a registry. Both come from
SKaiNET-embedded-vendors: `skainet/iree-compiler-torq:2.3.0` (`synaptics-torq/docker/build.sh`) and
`skainet/iree-runtime-torq:2.3.0` (`synaptics-torq/docker-runtime/build.sh`).

Target parameters (`blueprint.json` `targets[torq-sl26xx].params`): `capacity` (64), `weights` (`q8_0`), `head`
(`q4_0`), `conv` (`multiply-sum`: the short convolution as elementwise multiply + sum), `attention` (`batched`).
With all projections as Q4_0 the Torq 2.3.0 compiler fails to allocate local memory ("failed to allocate LRAM
addresses"); Q8_0 projections compile.

The plain commands, if you run the steps by hand:

```sh
./gradlew :llm-inference:lfm2:exportLfm2DecodeStep --args="--config <dir>/config.json --config-sha256 <sha> \
  --checkpoint <dir>/model.safetensors --checkpoint-sha256 <sha> --capacity 64 --output export"   # in SKaiNET-transformers
# retype: torq-bf16-retype (this repository's plugin, TorqBf16Retype)
docker run --rm -v "$PWD:/w" -w /w skainet/iree-compiler-torq:2.3.0 \
  compile-torq decode_step-bf16.mlir --out decode_step.vmfb --target SL2610 --input stablehlo-torq
```

The exporter is `sk.ainet.transformers:skainet-transformers-inference-lfm2` (`-Plfm2TorqTransformersVersion`,
default 0.64.2). To export with an unreleased exporter, pass `-Plfm2TransformersCheckout=/path/to/SKaiNET-transformers`.

## Reference measurement (an expectation, never copied into a descriptor)

On an SL2610 board with the Torq 2.3.0 runtime, one step at position 5 with a partly filled cache: one dispatch,
**533 ms** on the NPU, 274 ms to load the 350 MB module once; logits cosine 0.996 to the FP32 model, the same
top-1, 9 of the top 10 shared.
