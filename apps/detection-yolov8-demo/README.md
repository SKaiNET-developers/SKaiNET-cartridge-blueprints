# detection-yolov8-demo

A hands-on Android app for the [`detection-yolov8-iree`](../../blueprints/detection-yolov8-iree/) blueprint:
pick a photo, run a materialized `yolo/v1` pack on the phone, see labeled boxes and the latency. Every pack
you push gets a button — `cpu-arm64` and `vulkan-arm64` side by side is the CPU/GPU comparison, on the same
image, through the same public API (`YoloV8Cartridge.open` → `YoloEngine.infer`).

This is the blueprint's minimal smoke app. The full demo — live camera, Compose Multiplatform,
built against the published Maven artifacts — lives in
[SKaiNET-examples/VisionDemo](https://github.com/SKaiNET-developers/SKaiNET-examples/tree/develop/VisionDemo).

The repo ships **no weights and no compiled model** — the packs are AGPL-3.0 (see the blueprint's
*Licensing* section) and stay out of git. Instead, `assembleDebug` bundles whatever packs sit in the
blueprint's local `build/cartridge/` into the APK as assets, and the app unpacks them to its files dir on
first launch. So after materializing, the whole demo is:

```sh
./gradlew :apps:detection-yolov8-demo:installDebug
```

**Your build, your obligations**: the packs enter only your locally built APK (~43 MB with both; private
use on your own device). Handing that APK to anyone else is distribution of the AGPL weights — the
obligations travel with it, exactly as the blueprint's Licensing section and ADR-017 lay out.

Packs can also be loaded at runtime without rebuilding (scoped storage hides shell-created dirs under the
external root, so go through internal storage — debug builds only):

```sh
adb push <pack_dir> /data/local/tmp/yolo-packs/<pack_dir>
adb shell "run-as sk.ainet.cartridge.demo.yolo sh -c 'mkdir -p files/packs && cp -r /data/local/tmp/yolo-packs/* files/packs/'"
adb shell rm -rf /data/local/tmp/yolo-packs
```

Launch **YOLO Cartridge**, allow the camera, tap a pack — detection runs live on the camera feed, each box
labeled with class name and probability, the status line showing the per-inference latency (letterbox →
head → NMS → source pixels, the full engine path). Switching pack buttons mid-stream is the CPU/GPU
comparison. *Pick image* pauses the feed for a one-shot run on a gallery photo (*Run again* repeats it,
first run on a fresh pack is the cold one); *Live* resumes the feed.

The analysis stream is the preview (no CameraX `Preview` use case), so frames shown and frames analyzed
are the same bitmaps and the overlay maps 1:1. No network; camera is optional — without the permission the
picker path still works.
