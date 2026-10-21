package sk.ainet.cartridge.detection.yolov8

import kotlin.coroutines.cancellation.CancellationException
import sk.ainet.cartridge.profile.yolo.v1.BoundingBox
import sk.ainet.cartridge.profile.yolo.v1.Detection
import sk.ainet.cartridge.profile.yolo.v1.YoloEngine
import sk.ainet.cartridge.profile.yolo.v1.YoloEngineInfo
import sk.ainet.cartridge.profile.yolo.v1.YoloError
import sk.ainet.cartridge.profile.yolo.v1.YoloErrorCode
import sk.ainet.cartridge.profile.yolo.v1.YoloImage
import sk.ainet.cartridge.profile.yolo.v1.YoloImageFormat
import sk.ainet.cartridge.profile.yolo.v1.YoloResult
import sk.ainet.data.media.Letterbox
import sk.ainet.data.media.nonMaxSuppression

/**
 * Runs the cartridge's one compiled function: a letterboxed `[1, 3, size, size]` float image in, the raw head
 * `[1, 4 + classes, anchors]` out. The platform binding implements it over the IREE runtime; tests and the host
 * parity check implement it over SKaiNET or a fixture.
 */
public interface HeadRunner : AutoCloseable {
    public fun run(input: FloatArray): FloatArray
}

/**
 * The `yolo/v1` engine of this blueprint: RGB888 → letterbox → [HeadRunner] → candidates → class-aware NMS → boxes
 * in the source image's pixels. Everything model-specific about the output layout lives in [RawHead]; the
 * letterbox and NMS are SKaiNET's (`skainet-data-media`).
 */
public class YoloV8Engine(
    override val info: YoloEngineInfo,
    private val settings: DetectionSettings,
    private val runner: HeadRunner,
) : YoloEngine {

    public constructor(pack: PackDir, runner: HeadRunner) : this(
        info = YoloEngineInfo(
            cartridgeId = pack.id,
            name = "yolov8n",
            version = pack.version,
            labels = pack.settings.labels,
            inputResolution = pack.settings.inputResolution,
        ),
        settings = pack.settings,
        runner = runner,
    )

    private val anchors = RawHead.anchors(settings.inputResolution)

    override suspend fun infer(image: YoloImage): YoloResult {
        if (image.format != YoloImageFormat.RGB888 || image.width <= 0 || image.height <= 0 ||
            image.pixels.size != image.width * image.height * 3
        ) {
            return failed(YoloErrorCode.IMAGE_DECODE_FAILED, "expected ${image.width}x${image.height} RGB888 (${image.width * image.height * 3} bytes), got ${image.pixels.size} bytes", recoverable = true)
        }
        val letterbox = Letterbox.fit(image.width, image.height, settings.inputResolution)
        val head = try {
            runner.run(letterbox.fillNchw(image.pixels, settings.padValue))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            return failed(YoloErrorCode.INFERENCE_FAILED, e.message ?: e::class.simpleName ?: "inference failed", recoverable = true)
        }
        val numClasses = head.size / anchors - 4
        if (numClasses <= 0 || head.size % anchors != 0) {
            return failed(YoloErrorCode.INFERENCE_FAILED, "head of ${head.size} values does not fit $anchors anchors", recoverable = false)
        }
        val candidates = RawHead.decode(head, numClasses, anchors, settings.confThreshold)
        val kept = nonMaxSuppression(candidates, settings.iouThreshold, settings.maxDetections)
        return YoloResult.Detections(kept.map { d ->
            val b = letterbox.toSource(d.box)
            Detection(BoundingBox(b.x1, b.y1, b.x2, b.y2), d.classId, settings.labels.getOrNull(d.classId), d.score)
        })
    }

    override fun close(): Unit = runner.close()

    private fun failed(code: YoloErrorCode, message: String, recoverable: Boolean) =
        YoloResult.Failed(YoloError(code, message, recoverable))
}
