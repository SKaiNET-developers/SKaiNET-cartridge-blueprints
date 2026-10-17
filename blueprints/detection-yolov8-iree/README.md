# Blueprint: `detection-yolov8-iree`

Single-shot object detection: **YOLOv8n** (COCO, 80 classes, 640×640) as one SKaiNET-DSL graph, exported to
StableHLO and compiled with IREE for a Vulkan GPU or the CPU. An RGB image in, boxes in the image's own pixels out,
behind the [`yolo/v1`](../../profiles/yolo-v1/) task profile.

This directory is a [cartridge blueprint](https://skainet-developers.github.io/SKaiNET-cartridge/skainet-cartridge/blueprints.html):
the Kotlin API, the model graph and the complete build recipe — **no weights, no compiled model, no native
library**. You materialize it into a cartridge with *your* profile.

> **Status — work in progress.** The Kotlin API, the post-processing and the `fetch`, `convert`, `export`,
> `parameters` and `compile` steps run and are tested; the exported graph matches ONNX Runtime on the host. The
> runtime (`IreeModuleRunner` in SKaiNET-transformers) is not released yet, so a cartridge cannot be materialized
> today.

| | |
|---|---|
| Recipe | [`blueprint.json`](blueprint.json) |
| Weights | `Ultralytics/YOLOv8` @ `27f858f2…`: `yolov8n.pt`, pinned by SHA-256 |
| Weights license | **AGPL-3.0** at the publisher (model card `license: agpl-3.0`, verified 2026-10-16). `redistribution: restricted` — see *Licensing* before you ship a cartridge |
| Code license | MIT |
| Kotlin API | Kotlin Multiplatform — `android`, `jvm`, `linuxX64`, `linuxArm64`, `macosArm64`. Engine, post-processing, pack locator and the YOLOv8 graph are common code; the Android runtime binding lands with `IreeModuleRunner` |
| Targets | `vulkan-arm64`, `cpu-arm64` (Android `arm64-v8a`) |
| Toolchain | SKaiNET 0.60.1, Ultralytics 8.4.170 (convert only), IREE tools 3.11.0, SKaiNET-transformers runtime 0.61.1 |

## What you will build

A signed cartridge `pack_dir` holding one compiled graph (`artifacts/model/yolov8n.vmfb`), its parameter archive
(`artifacts/weights/params.irpa`) and the generic module runner for the target ABI, plus a descriptor that states
what was measured on *your* device.

The cartridge is a plain tensor function: `float32 [1, 3, 640, 640]` (RGB, `[0, 1]`, letterboxed) in,
`float32 [1, 84, 8400]` out — rows 0–3 box center/size in input pixels, rows 4–83 per-class scores. Everything
image-shaped happens in the Kotlin adapter, `YoloV8Engine`:

```
RGB888 ─► Letterbox.fit / fillNchw ─► HeadRunner (the cartridge) ─► RawHead.decode ─► nonMaxSuppression ─► Letterbox.toSource ─► yolo/v1 Detections
          (skainet-data-media)                                         (this blueprint)    (skainet-data-media)
```

Thresholds, labels and the input size come from the descriptor's `attributes` (`conf_threshold`, `iou_threshold`,
`max_detections`, `label_set`, `input_resolution`), not from host code.

## Prerequisites

- JDK 21+, Git; Android SDK (`ANDROID_HOME`) with platform 36 — one of the API's targets is Android.
- Docker, for the `convert` step (Ultralytics exporter) and the IREE tools steps.

## Step 1 — Profile

Copy [`profiles/example.cpu-arm64.json`](profiles/example.cpu-arm64.json) to `profiles/mine.json` and pick your
`target`. Leave `measurements` out until you have measured.

## Step 2 — Fetch and convert

```bash
./gradlew :blueprints:detection-yolov8-iree:convertOnnx -Pprofile=profiles/mine.json
```

`blueprintFetch` downloads `yolov8n.pt` at the pinned revision and checks its SHA-256. `convertOnnx` runs the
Ultralytics exporter in the pinned container (`yolo export format=onnx imgsz=640 opset=17`) and writes
`build/blueprint/work/convert/yolov8n.onnx`. Only its initializers — Conv weights with BatchNorm already fused —
are used downstream; the graph that gets compiled is the SKaiNET one in `src/commonMain/.../model/YoloV8.kt`. The ONNX
file's bytes differ between runs (the exporter stamps a date into its metadata), the initializers do not.

Plain equivalent:

```bash
docker run --rm -v "$PWD/build/blueprint/sources/weights:/in:ro" -v "$PWD/out:/out" -w /out \
  ultralytics/ultralytics:8.4.170-cpu \
  sh -c 'pip install --target /out/.pydeps onnx==1.23.2 && cp /in/yolov8n.pt . &&
         PYTHONPATH=/out/.pydeps yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17 simplify=False'
```

## Step 3 — Export

```bash
./gradlew :blueprints:detection-yolov8-iree:exportStableHlo -Pprofile=profiles/mine.json
```

`YoloV8ExportCli` (this module's `jvmExport` compilation) bakes the 127 ONNX initializers into the SKaiNET
`YoloV8` graph by name — every graph parameter must find its initializer with the exact shape, and every
initializer must be used — traces it and lowers it to StableHLO: `build/blueprint/work/export/yolov8n.mlir`, one
function `@yolov8_detect(tensor<1x3x640x640xf32>) -> tensor<1x84x8400xf32>`, weights as constants. Next to it,
`manifest.json` records the shapes and a SHA-256 over the initializers.

The graph is the Ultralytics export-mode `Detect` head in full: DFL softmax + projection, anchor-point decode,
stride scaling and class sigmoid run in-graph, so the cartridge's output is the decoded head the adapter reads.

## Step 4 — Parameters and compile

```bash
./gradlew :blueprints:detection-yolov8-iree:parameters :blueprints:detection-yolov8-iree:compile -Pprofile=profiles/mine.json
```

`parameters` moves the weights into `params.irpa` (scope `model`); `compile` builds `yolov8n.vmfb` for the
profile's target, referencing the archive.

## Steps 5–6 — Runtime, materialize *(pending)*

`extractRuntime` (`libskainet_iree_module.so` from the SKaiNET-transformers AAR) and `materializeCartridge` run
once the runtime is released.

## Use from an app

```kotlin
val pack = PackDir("/data/.../detection-yolov8-iree-cpu-arm64")
val engine: YoloEngine = YoloV8Engine(pack, runner)        // runner: the platform's HeadRunner over the cartridge
when (val r = engine.infer(YoloImage(rgb888, width, height))) {
    is YoloResult.Detections -> r.boxes.forEach { println("${it.label} ${it.score} ${it.box.x1},${it.box.y1}") }
    is YoloResult.Failed -> println(r.error.message)
}
```

## Licensing

The weights are AGPL-3.0. A cartridge materialized from them is a derived work of the weights: whoever ships one
— in an app, on a device, or behind a network service — takes on the AGPL obligations (source of the whole work
and network-use clause). The blueprint marks the source `restricted` so the materializer warns, and never accepts
anything on your behalf. Ultralytics offers a separate enterprise license for uses that cannot meet the AGPL; that
is a matter between you and them, recorded in your profile, not in this blueprint.

## Reproducibility

Every input is pinned: the checkpoint by revision and SHA-256, the Ultralytics container by digest, `onnx` by
version, IREE tools by image tag, SKaiNET and the runtime by version.
