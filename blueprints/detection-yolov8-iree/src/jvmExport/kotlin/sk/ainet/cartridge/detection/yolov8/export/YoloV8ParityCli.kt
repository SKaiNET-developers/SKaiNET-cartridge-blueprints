package sk.ainet.cartridge.detection.yolov8.export

import sk.ainet.cartridge.detection.yolov8.RawHead
import sk.ainet.cartridge.detection.yolov8.model.YoloV8
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.data.media.Letterbox
import sk.ainet.data.media.ScoredBox
import sk.ainet.data.media.nonMaxSuppression
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.system.exitProcess

/**
 * Host parity of the SKaiNET **eager** forward pass against an ONNX Runtime reference — the model half of B4; the
 * compiled halves (ORT vs IREE llvm-cpu on the host, IREE on the device) have their own checks. Same checkpoint,
 * same graph, same input as the export: the ONNX initializers baked into [YoloV8], run on
 * [DirectCpuExecutionContext] instead of being traced.
 *
 *   YOLO_ONNX=<yolov8n.onnx> YOLO_IMAGE=<image> YOLO_WORK=<dir> java … YoloV8ParityCliKt
 *
 * Writes into `YOLO_WORK` (raw little-endian float32, no header):
 *  - `input.f32` — the image letterboxed by the engine's own [Letterbox.fillNchw] `[1, 3, S, S]`; feed this to the
 *    reference runtime so both sides see identical bytes,
 *  - `eager-head.f32` — the eager head `[1, 4 + C, A]`, reused on the next run if present.
 *
 * With `<work>/ort-head.f32` present (or `YOLO_ORT_HEAD` set), compares head values at rtol = atol = 1e-2 and the
 * post-NMS detections (same classes in order, IoU ≥ 0.99, score Δ ≤ 1e-3), and exits non-zero on disagreement;
 * without it, only the eager artifacts are produced.
 */
public fun main() {
    val onnx = File(System.getenv("YOLO_ONNX") ?: error("set YOLO_ONNX to the Ultralytics ONNX export (yolov8n.onnx)"))
    val image = File(System.getenv("YOLO_IMAGE") ?: error("set YOLO_IMAGE to the test image (e.g. bus.jpg)"))
    val work = File(System.getenv("YOLO_WORK") ?: "build/parity").apply { mkdirs() }
    val inputSize = System.getenv("YOLO_INPUT_SIZE")?.toInt() ?: 640

    // The input, exactly as YoloV8Engine prepares it.
    val decoded = ImageIO.read(image) ?: error("cannot decode ${image.path}")
    val rgb = ByteArray(decoded.width * decoded.height * 3)
    for (y in 0 until decoded.height) for (x in 0 until decoded.width) {
        val p = decoded.getRGB(x, y)
        val i = (y * decoded.width + x) * 3
        rgb[i] = (p shr 16 and 0xFF).toByte(); rgb[i + 1] = (p shr 8 and 0xFF).toByte(); rgb[i + 2] = (p and 0xFF).toByte()
    }
    val letterbox = Letterbox.fit(decoded.width, decoded.height, inputSize)
    val input = letterbox.fillNchw(rgb)
    writeF32(File(work, "input.f32"), input)
    log("${image.name}: ${decoded.width}x${decoded.height} -> input.f32 [1, 3, $inputSize, $inputSize]")

    val checkpoint = OnnxInitializers(onnx)
    val config = checkpoint.config(inputSize)
    val anchors = config.anchors
    val rows = 4 + config.numClasses

    val eagerFile = File(work, "eager-head.f32")
    val eager: FloatArray
    if (eagerFile.exists()) {
        eager = readF32(eagerFile, rows * anchors)
        log("eager-head.f32 reused (delete it to recompute)")
    } else {
        val model = YoloV8(config)
        val ctx = DirectCpuExecutionContext.create()
        checkpoint.bake(model, ctx)
        log("baked ${checkpoint.names.size} initializers; eager forward on ${ctx.schedule.name} …")
        val t0 = System.nanoTime()
        val head = model.forward(ctx.fromFloatArray(Shape(1, 3, inputSize, inputSize), FP32::class, input), ctx)
        val dims = head.shape.dimensions.toList()
        check(dims == listOf(1, rows, anchors)) { "eager head $dims, expected [1, $rows, $anchors]" }
        eager = FloatArray(rows * anchors)
        for (r in 0 until rows) for (a in 0 until anchors) eager[r * anchors + a] = head.data.get(0, r, a)
        writeF32(eagerFile, eager)
        log("eager head in %.1f s -> eager-head.f32".format((System.nanoTime() - t0) / 1e9))
    }

    val detections = { head: FloatArray ->
        nonMaxSuppression(RawHead.decode(head, config.numClasses, anchors, CONF), IOU, MAX_DETECTIONS)
            .map { ScoredBox(letterbox.toSource(it.box), it.classId, it.score) }
    }
    val eagerBoxes = detections(eager)
    log("eager: " + describe(eagerBoxes))

    val ortFile = File(System.getenv("YOLO_ORT_HEAD") ?: File(work, "ort-head.f32").path)
    if (!ortFile.exists()) {
        log("no reference at ${ortFile.path} — run ONNX Runtime on input.f32, write the head there, rerun")
        return
    }
    val ort = readF32(ortFile, rows * anchors)
    var worst = 0f
    var outside = 0
    for (i in eager.indices) {
        val d = abs(eager[i] - ort[i])
        if (d > worst) worst = d
        if (d > ATOL + RTOL * abs(ort[i])) outside++
    }
    log("head vs ORT: max|Δ| %.3g, $outside/${eager.size} outside rtol/atol $RTOL".format(worst))

    val ortBoxes = detections(ort)
    log("ort:   " + describe(ortBoxes))
    val boxFailures = buildList {
        if (eagerBoxes.size != ortBoxes.size) add("${eagerBoxes.size} detections vs ORT's ${ortBoxes.size}")
        else eagerBoxes.zip(ortBoxes).forEachIndexed { i, (e, o) ->
            if (e.classId != o.classId) add("detection $i: class ${e.classId} vs ${o.classId}")
            else if (e.box.iou(o.box) < 0.99f) add("detection $i: IoU %.4f".format(e.box.iou(o.box)))
            else if (abs(e.score - o.score) > 1e-3f) add("detection $i: score Δ %.4g".format(abs(e.score - o.score)))
        }
    }
    if (outside > 0 || boxFailures.isNotEmpty()) {
        log("PARITY FAILED" + boxFailures.joinToString("") { "\n  $it" })
        exitProcess(1)
    }
    log("PARITY OK: head within tolerance, ${eagerBoxes.size} identical post-NMS detections")
}

private const val CONF = 0.25f
private const val IOU = 0.45f
private const val MAX_DETECTIONS = 300
private const val RTOL = 1e-2f
private const val ATOL = 1e-2f

private fun describe(boxes: List<ScoredBox>) =
    "${boxes.size} detections " + boxes.joinToString { "c${it.classId} %.3f".format(it.score) }

private fun log(message: String) = println("[yolov8-parity] $message")

private fun writeF32(file: File, values: FloatArray) {
    val bytes = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    bytes.asFloatBuffer().put(values)
    file.writeBytes(bytes.array())
}

private fun readF32(file: File, expected: Int): FloatArray {
    val bytes = file.readBytes()
    require(bytes.size == expected * 4) { "${file.path}: ${bytes.size} bytes, expected $expected float32 values" }
    return FloatArray(expected).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
}
