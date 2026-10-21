package sk.ainet.cartridge.profile.yolo.v1

/**
 * Task profile `yolo/v1` — the public, versioned consumer contract for YOLO-family
 * object-detection cartridges (spec: docs/modules/ROOT/pages/task-profiles.adoc, ADR-008).
 *
 * The profile owns ALL of its types: nothing here references any application's contract,
 * any specific YOLO version/weights, or any one cartridge's io shape. Capability
 * differences between engines (input resolution, class list) stay visible in
 * [YoloEngineInfo] and the cartridge descriptor — the profile hides the io shape, never
 * the capability.
 */

/** Profile-owned engine description — mirrors descriptor facts, never app concepts. */
public class YoloEngineInfo(
    /** The cartridge id this engine wraps (descriptor `id`), if cartridge-backed. */
    public val cartridgeId: String?,
    public val name: String,
    public val version: String,
    /** The cartridge's declared class list, in class-id order (descriptor `attributes.label_set`). */
    public val labels: List<String>,
    /** Expected square input side in pixels (descriptor `attributes.input_resolution`). */
    public val inputResolution: Int,
)

/**
 * One image to run detection on. Raw decoded pixels on purpose, same rationale as
 * `sk.ainet.cartridge.profile.asr.v1.AsrAudioChunk`: no encoding ambiguity at the API boundary — the
 * implementation owns resizing/letterboxing/normalization into the cartridge's io shape,
 * and undoes it again on the way out (see [BoundingBox]).
 */
public class YoloImage(
    public val pixels: ByteArray,
    public val width: Int,
    public val height: Int,
    /** One byte per channel, row-major, no padding — the only layout v1 supports. */
    public val format: YoloImageFormat = YoloImageFormat.RGB888,
)

public enum class YoloImageFormat { RGB888 }

public sealed interface YoloResult {
    /** Terminal success. Empty list is a valid result (nothing detected), not an error. */
    public data class Detections(public val boxes: List<Detection>) : YoloResult

    /** Terminal failure. */
    public data class Failed(public val error: YoloError) : YoloResult
}

/** One detected object. */
public class Detection(
    public val box: BoundingBox,
    public val classId: Int,
    /**
     * Resolved from the cartridge's `attributes.label_set` when available; null if the
     * engine can't resolve a name for [classId] (e.g. an out-of-range index, or a
     * cartridge that doesn't declare `label_set`). Consumers needing a guaranteed name
     * should fall back to [YoloEngineInfo.labels]`.getOrNull(classId)` themselves.
     */
    public val label: String? = null,
    public val score: Float,
)

/**
 * Axis-aligned box, pixel-space, top-left origin, in the ORIGINAL image's coordinates —
 * not the model's resized input. The implementation undoes any letterbox/resize before
 * returning, so consumers never need to know the model's native resolution.
 */
public class BoundingBox(
    public val x1: Float,
    public val y1: Float,
    public val x2: Float,
    public val y2: Float,
) {
    public val width: Float get() = (x2 - x1).coerceAtLeast(0f)
    public val height: Float get() = (y2 - y1).coerceAtLeast(0f)
}

public class YoloError(
    public val code: YoloErrorCode,
    public val message: String,
    public val recoverable: Boolean,
)

public enum class YoloErrorCode {
    ENGINE_UNAVAILABLE,
    MODEL_LOAD_FAILED,
    IMAGE_DECODE_FAILED,
    INFERENCE_FAILED,
    INFERENCE_TIMEOUT,
    CANCELLED,
    UNKNOWN,
}
