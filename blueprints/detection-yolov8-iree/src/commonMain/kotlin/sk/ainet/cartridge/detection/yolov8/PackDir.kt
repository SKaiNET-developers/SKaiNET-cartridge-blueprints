package sk.ainet.cartridge.detection.yolov8

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * A materialized YOLOv8 detection cartridge on disk (cartridge spec, ADR-012): `descriptor.json`, `manifest.json` and
 * `artifacts/<role>/…` as `materializeCartridge` wrote them. Locates things only — verifying the manifest before load
 * is the host's job, per the spec's threat model.
 */
public class PackDir(public val root: Path) {
    public constructor(root: String) : this(Path(root))

    init {
        require(SystemFileSystem.exists(Path(root, "descriptor.json"))) { "not a cartridge pack_dir (no descriptor.json): $root" }
    }

    public val descriptor: JsonObject by lazy { Json.parseToJsonElement(readText(Path(root, "descriptor.json"))).jsonObject }

    public val id: String get() = descriptor.getValue("id").jsonPrimitive.content

    public val version: String get() = descriptor["version"]?.jsonPrimitive?.content ?: "unknown"

    /** `true` when the descriptor's target names an accelerator — the cartridge was built for a GPU. */
    public val targetsAccelerator: Boolean get() = descriptor["target"]?.jsonObject?.containsKey("accelerator") == true

    /** Decode settings and labels from descriptor `attributes`; the defaults are the blueprint's. */
    public val settings: DetectionSettings by lazy {
        val a = descriptor["attributes"]?.jsonObject
        val d = DetectionSettings()
        DetectionSettings(
            labels = a?.get("label_set")?.jsonArray?.map { it.jsonPrimitive.content } ?: d.labels,
            inputResolution = a?.get("input_resolution")?.jsonPrimitive?.int ?: d.inputResolution,
            confThreshold = a?.get("conf_threshold")?.jsonPrimitive?.float ?: d.confThreshold,
            iouThreshold = a?.get("iou_threshold")?.jsonPrimitive?.float ?: d.iouThreshold,
            maxDetections = a?.get("max_detections")?.jsonPrimitive?.int ?: d.maxDetections,
        )
    }

    /** The compiled graph and its parameter archive. */
    public fun model(): ModelFiles = ModelFiles(
        vmfb = file("artifacts/model/yolov8n.vmfb"),
        paramsIrpa = file("artifacts/weights/params.irpa"),
    )

    public data class ModelFiles(val vmfb: Path, val paramsIrpa: Path)

    private fun file(relative: String): Path = Path(root, relative).also {
        require(SystemFileSystem.exists(it)) { "pack_dir ${root.name} is missing $relative" }
    }

    internal companion object {
        fun readText(path: Path): String = SystemFileSystem.source(path).buffered().use { it.readString() }
    }
}
