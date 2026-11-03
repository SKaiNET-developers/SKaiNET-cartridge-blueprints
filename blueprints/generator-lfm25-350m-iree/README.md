# LFM2.5-350M complete Vulkan generation candidate

This source recipe builds the complete cached hybrid decoder at a compiled capacity of **64 tokens**. It traces SKaiNET-transformers to StableHLO, converts external FP32 parameters to IRPA, compiles generic Vulkan through SKaiNET-IREE-tools, builds the shared `llm/v1` JVM worker and native C ABI bridge, and stages three pinned Python 3.12 ARM64 wheels. Token selection, RoPE/mask controls and stopping run on the host; model operations and hybrid state run on the explicit Vulkan device. There is no CPU model fallback.

**GPU target qualification is pending.** The first external full-model compile exposed an oversized GPU initializer binding despite64MiB source embedding shards. The current recipe uses eight8192-row source embedding shards, explicit resource bounds with binding fusion disabled, and a fail-closed audit of the exact compiled VMFB's HAL bindings, allocations and dispatch grids before creating its runtime manifest. This static audit does not prove device execution or numerical correctness. The source pipeline and provisional directory do not authorize materialization. `materializeCartridge` deliberately fails until independent complete-model numerical/state/generation validation, frontend generation, C ABI behavior and exact-artifact performance evidence are implemented and accepted. No attention-only evidence, previous CPU timings, synthetic metrics, license acceptance or signing keys can satisfy that gate.

The adapter, worker, native bridge and tests are shared in `:runtimes:lfm2-jvm`. Actual profile types come from `:profiles:llm-v1`. The model and runtime come from the two new SKaiNET-transformers dependencies; while unpublished, supply an independent checkout with `-Plfm2TransformersCheckout=/path/to/SKaiNET-transformers`. A committed immutable dependency/release pin remains required before a public release.

Create a private settings file outside the source tree:

```json
{
  "model_directory": "/absolute/path/to/verified/LFM2.5-350M",
  "wheel_directory": "/absolute/path/to/verified/python312-aarch64-wheels",
  "device_uri": "vulkan://YOUR_EXPLICIT_DEVICE"
}
```

The seven checkpoint pins are in `blueprint.json`; `runtime-wheels.json` pins the exact IREE 3.11.0, NumPy and ml_dtypes wheels. The recipe verifies local inputs and the actual source fetcher uses those directories as private mirrors. It extracts wheel members without running installers; paths, duplicates, symlinks and overwrite attempts fail closed. No Python interpreter, Java runtime, weights or compiled files belong in the repository.
The wheel sources remain `NOASSERTION`: their top-level package labels do not describe every bundled library. For example, the pinned NumPy wheel includes OpenBLAS and GCC runtime notices. Extraction preserves all included metadata/license files; complete dependency license review remains a release gate.

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

Outputs live under `build/runs/<gpuRun>/`. Preserve every failed attempt and choose a new run ID. When changing only compiler policy, `-Plfm2PreparedExport=/absolute/path/to/prior/export/bundle` reuses a captured source export after rehashing its full module/parameter/contract closure. It imports neither a compiled binary nor accepted measurements, and avoids duplicating the large FP32 parameter file. The provisional pack contains VMFB, IRPA, source/build provenance, six tokenizer/config/license files, private Python packages, shared worker JARs and the real native library. It omits the original BF16 checkpoint duplicate. `execution.json` records every actual file hash and explicitly records pending GPU gates. `manifest.json` is unsigned staging metadata, not a signed Cartridge Spec manifest.

Deployment requires external `JAVA_HOME` (Java 21 with `jdk.incubator.vector`) and `LFM2_PYTHON_EXECUTABLE` (Python 3.12), plus the selected Vulkan driver/device. The shared worker's preflight checks Python/IREE/Vulkan without loading IRPA/model weights. The worker advertises the compiled capacity64 and rejects excess prompt/output budgets; the source checkpoint's2048-token maximum is not the GPU runtime capacity.
