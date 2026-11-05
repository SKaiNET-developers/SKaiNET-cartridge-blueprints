# LFM2.5-350M complete Vulkan generation candidate

This source recipe builds the complete cached hybrid decoder at a compiled capacity of **64 tokens**. It traces SKaiNET-transformers to StableHLO, converts external FP32 parameters to IRPA, compiles generic Vulkan through SKaiNET-IREE-tools, builds the shared `llm/v1` JVM worker and native C ABI bridge, and stages a **source-built native IREE runtime** (`skainet/iree-linux-arm64`: aarch64 glibc tools, the upstream license set and a provenance report) — there is no Python on the target. Token selection, RoPE/mask controls and stopping run on the host; model operations and hybrid state run on the explicit Vulkan device. There is no CPU model fallback. The shared JVM worker's native IREE backend is a pending gate: the staged worker configuration (`backend: iree-native-vulkan`) fails closed until that backend is implemented and qualified; the earlier Python-worker lane is retired from the pack.

**GPU target qualification is pending.** The first external full-model compile exposed an oversized GPU initializer binding despite64MiB source embedding shards. The current recipe uses eight8192-row source embedding shards, explicit resource bounds with binding fusion disabled, and a fail-closed audit of the exact compiled VMFB's HAL bindings, allocations and dispatch grids before creating its runtime manifest. This static audit does not prove device execution or numerical correctness. The source pipeline and provisional directory do not authorize materialization. `materializeCartridge` deliberately fails until independent complete-model numerical/state/generation validation, frontend generation, C ABI behavior and exact-artifact performance evidence are implemented and accepted. No attention-only evidence, previous CPU timings, synthetic metrics, license acceptance or signing keys can satisfy that gate.

The adapter, worker, native bridge and tests are shared in `:runtimes:lfm2-jvm`. Actual profile types come from `:profiles:llm-v1`. The model and runtime come from the two new SKaiNET-transformers dependencies; while unpublished, supply an independent checkout with `-Plfm2TransformersCheckout=/path/to/SKaiNET-transformers`. A committed immutable dependency/release pin remains required before a public release.

Create a private settings file outside the source tree:

```json
{
  "model_directory": "/absolute/path/to/verified/LFM2.5-350M",
  "device_uri": "vulkan://YOUR_EXPLICIT_DEVICE"
}
```

The seven checkpoint pins are in `blueprint.json`. The recipe verifies local inputs and the actual source fetcher uses the model directory as a private mirror. No Python interpreter, Java runtime, weights or compiled files belong in the repository.

The native IREE runtime replaces the three Python runtime wheels of the earlier revision (the wheels brought a six-term NumPy compound license including `GPL-3.0-or-later WITH GCC-exception-3.1`, and the IREE wheel shipped no license texts at all). The runtime is now source-built from the pinned IREE tree (`iree-runtime-source`, tag `v3.11.0`, commit `e4a3b04`) through the `skainet/iree-linux-arm64` toolchain image: `Apache-2.0 WITH LLVM-exception`, with the upstream license texts and a provenance report (git commit, HAL drivers, per-file SHA-256) staged next to the binaries in `artifacts/iree/`, resolving the earlier notice-text gap. `prepare` re-verifies the report pins and every file digest before and after staging. The blueprint's own license stays `NOASSERTION` until the recipe is licensed, so the computed effective license remains `NOASSERTION` by design.

From the repository root:

```sh
./gradlew :blueprints:generator-lfm25-350m-iree:check \
  -Plfm2TransformersCheckout=/path/to/SKaiNET-transformers

./gradlew :blueprints:generator-lfm25-350m-iree:prepareGpuExecution \
  -Plfm2TransformersCheckout=/path/to/SKaiNET-transformers \
  -Plfm2IreeToolsCheckout=/path/to/SKaiNET-IREE-tools \
  -Plfm2GpuSettings=/private/path/settings.json -PgpuRun=001
```

The source-mounted `docker/compiler/entrypoint.sh` is essential: an older compiler image entrypoint discards compiler options and selects a device-specific default. The recipe inspects the local IREE 3.11.0 image, executes its immutable ID with networking disabled, copies the selected wrapper into the compilation evidence, and passes explicit `generic`, `stablehlo` and no-FP16-demotion options. It does not claim that a modified image has been published.

Outputs live under `build/runs/<gpuRun>/`. Preserve every failed attempt and choose a new run ID. When changing only compiler policy, `-Plfm2PreparedExport=/absolute/path/to/prior/export/bundle` reuses a captured source export after rehashing its full module/parameter/contract closure. It imports neither a compiled binary nor accepted measurements, and avoids duplicating the large FP32 parameter file. `-Plfm2IreeRuntime=/absolute/path/to/prior/iree-runtime/out` similarly reuses a prior native-runtime build; its report pins and file digests are re-verified either way. The provisional pack contains VMFB, IRPA, source/build provenance, six tokenizer/config/license files, the staged native IREE runtime (binaries, license set, provenance report), shared worker JARs and the real native library. It omits the original BF16 checkpoint duplicate. `execution.json` records every actual file hash and explicitly records pending GPU gates. `manifest.json` is unsigned staging metadata, not a signed Cartridge Spec manifest.

Deployment requires external `JAVA_HOME` (Java 21 with `jdk.incubator.vector`) plus the selected Vulkan driver/device; the IREE runtime ships inside the pack and no Python is required on the target. The worker advertises the compiled capacity64 and rejects excess prompt/output budgets; the source checkpoint's2048-token maximum is not the GPU runtime capacity. Until the shared worker's native IREE backend lands, the staged `worker.json` is rejected by the current worker — deliberately fail-closed, like every other pending gate of this recipe.
