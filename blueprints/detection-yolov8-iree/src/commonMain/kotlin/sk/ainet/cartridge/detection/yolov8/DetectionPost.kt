package sk.ainet.cartridge.detection.yolov8

import sk.ainet.data.media.BoundingBox
import sk.ainet.data.media.ScoredBox

/**
 * What the adapter needs to turn the cartridge's raw head into `yolo/v1` detections; read from descriptor
 * `attributes` by [PackDir.settings].
 */
public data class DetectionSettings(
    val labels: List<String> = emptyList(),
    val inputResolution: Int = 640,
    val confThreshold: Float = 0.25f,
    val iouThreshold: Float = 0.45f,
    val maxDetections: Int = 300,
    /** Letterbox padding, 0..255 per channel; 114 is what the checkpoint was trained with. */
    val padValue: Int = 114,
)

/**
 * The cartridge's output tensor `[1, 4 + classes, anchors]` (descriptor `attributes.output_layout`): rows 0–3 are box
 * center x, center y, width and height in input-canvas pixels, the remaining rows per-class scores, already sigmoid.
 */
public object RawHead {
    /** Number of anchors for a square [inputResolution] over strides 8, 16 and 32 (8400 at 640). */
    public fun anchors(inputResolution: Int): Int =
        listOf(8, 16, 32).sumOf { s -> (inputResolution / s) * (inputResolution / s) }

    /**
     * Candidates at or above [confThreshold], one per anchor (its best class), in canvas pixels. No NMS here.
     */
    public fun decode(head: FloatArray, numClasses: Int, anchors: Int, confThreshold: Float): List<ScoredBox> {
        require(head.size == (4 + numClasses) * anchors) {
            "head has ${head.size} values, expected (4 + $numClasses) x $anchors = ${(4 + numClasses) * anchors}"
        }
        val out = ArrayList<ScoredBox>()
        for (a in 0 until anchors) {
            var best = -1
            var bestScore = Float.NEGATIVE_INFINITY
            for (c in 0 until numClasses) {
                val s = head[(4 + c) * anchors + a]
                if (s > bestScore) { bestScore = s; best = c }
            }
            if (bestScore < confThreshold) continue
            val box = BoundingBox.fromCenter(head[a], head[anchors + a], head[2 * anchors + a], head[3 * anchors + a])
            out += ScoredBox(box, best, bestScore)
        }
        return out
    }
}
