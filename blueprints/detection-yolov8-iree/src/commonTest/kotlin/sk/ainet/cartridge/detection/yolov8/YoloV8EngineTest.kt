package sk.ainet.cartridge.detection.yolov8

import kotlinx.coroutines.test.runTest
import sk.ainet.cartridge.profile.yolo.v1.YoloEngineInfo
import sk.ainet.cartridge.profile.yolo.v1.YoloErrorCode
import sk.ainet.cartridge.profile.yolo.v1.YoloImage
import sk.ainet.cartridge.profile.yolo.v1.YoloResult
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class YoloV8EngineTest {
    // A small canvas keeps the synthetic head readable: 64 px → 8x8 + 4x4 + 2x2 = 84 anchors, 2 classes.
    private val size = 64
    private val classes = 2
    private val anchors = RawHead.anchors(size)
    private val settings = DetectionSettings(labels = listOf("person", "cat"), inputResolution = size)

    /** A head with the given (cx, cy, w, h, class, score) entries on the first anchors, everything else zero. */
    private fun head(vararg entries: List<Float>): FloatArray {
        val h = FloatArray((4 + classes) * anchors)
        entries.forEachIndexed { a, e ->
            for (r in 0 until 4) h[r * anchors + a] = e[r]
            h[(4 + e[4].toInt()) * anchors + a] = e[5]
        }
        return h
    }

    private class FixedRunner(private val head: FloatArray) : HeadRunner {
        var input: FloatArray? = null
        var closed = false
        override fun run(input: FloatArray): FloatArray { this.input = input; return head }
        override fun close() { closed = true }
    }

    private fun engine(runner: HeadRunner) =
        YoloV8Engine(YoloEngineInfo("test", "yolov8n", "0.1.0", settings.labels, size), settings, runner)

    private fun image(w: Int, h: Int) = YoloImage(ByteArray(w * h * 3) { 128.toByte() }, w, h)

    @Test
    fun anchorsAt640AreThe8400OfTheReleasedHead() {
        assertEquals(8400, RawHead.anchors(640))
    }

    @Test
    fun boxesComeBackInSourcePixelsWithLabels() = runTest {
        // 128x64 source → scale 0.5, resized 64x32, padTop 16. A canvas box centered (32, 32), 32x16 → source (32, 16)..(96, 48).
        val runner = FixedRunner(head(listOf(32f, 32f, 32f, 16f, 1f, 0.9f)))
        val result = assertIs<YoloResult.Detections>(engine(runner).infer(image(128, 64)))
        val d = result.boxes.single()
        assertEquals(1, d.classId)
        assertEquals("cat", d.label)
        assertEquals(0.9f, d.score)
        assertEquals(listOf(32f, 16f, 96f, 48f), listOf(d.box.x1, d.box.y1, d.box.x2, d.box.y2))
        assertEquals(3 * size * size, runner.input!!.size)
    }

    @Test
    fun overlappingBoxesOfOneClassAreSuppressedAndOtherClassesKept() = runTest {
        val runner = FixedRunner(head(
            listOf(20f, 20f, 10f, 10f, 0f, 0.8f),
            listOf(21f, 21f, 10f, 10f, 0f, 0.95f),
            listOf(20f, 20f, 10f, 10f, 1f, 0.7f),
        ))
        val result = assertIs<YoloResult.Detections>(engine(runner).infer(image(size, size)))
        assertEquals(listOf(0.95f to 0, 0.7f to 1), result.boxes.map { it.score to it.classId })
    }

    @Test
    fun belowThresholdIsAnEmptyResultNotAnError() = runTest {
        val runner = FixedRunner(head(listOf(20f, 20f, 10f, 10f, 0f, 0.1f)))
        val result = assertIs<YoloResult.Detections>(engine(runner).infer(image(size, size)))
        assertTrue(result.boxes.isEmpty())
    }

    @Test
    fun aMalformedImageIsAnImageDecodeFailure() = runTest {
        val result = assertIs<YoloResult.Failed>(engine(FixedRunner(head())).infer(YoloImage(ByteArray(5), 2, 2)))
        assertEquals(YoloErrorCode.IMAGE_DECODE_FAILED, result.error.code)
    }

    @Test
    fun aRunnerFailureIsAnInferenceFailure() = runTest {
        val failing = object : HeadRunner {
            override fun run(input: FloatArray): FloatArray = error("device lost")
            override fun close() {}
        }
        val result = assertIs<YoloResult.Failed>(engine(failing).infer(image(size, size)))
        assertEquals(YoloErrorCode.INFERENCE_FAILED, result.error.code)
        assertEquals("device lost", result.error.message)
    }

    @Test
    fun aHeadOfTheWrongSizeIsReported() = runTest {
        val result = assertIs<YoloResult.Failed>(engine(FixedRunner(FloatArray(7))).infer(image(size, size)))
        assertEquals(YoloErrorCode.INFERENCE_FAILED, result.error.code)
    }

    @Test
    fun closeReleasesTheRunner() {
        val runner = FixedRunner(head())
        engine(runner).close()
        assertTrue(runner.closed)
    }
}
