package sk.ainet.cartridge.detection.yolov8

import sk.ainet.cartridge.detection.yolov8.model.YoloV8
import sk.ainet.cartridge.detection.yolov8.model.YoloV8Config
import kotlin.test.Test
import kotlin.test.assertEquals

class YoloV8GraphTest {
    /** `name -> shape` of the Ultralytics YOLOv8n ONNX export, from the test resource. */
    private val checkpoint: Map<String, List<Int>> =
        requireNotNull(javaClass.getResource("/yolov8n-initializers.txt")).readText().lines()
            .filter { it.isNotBlank() && !it.startsWith("#") }
            .associate { line -> line.substringBefore(' ') to line.substringAfter(' ').split('x').map(String::toInt) }

    @Test
    fun parametersAreExactlyTheCheckpointsInitializers() {
        val graph = YoloV8().parameters().associate { it.name to it.value.shape.dimensions.toList() }
        assertEquals(checkpoint.keys.sorted(), graph.keys.sorted())
        for ((name, shape) in checkpoint) assertEquals(shape, graph[name], name)
    }

    @Test
    fun yolov8nGeometry() {
        val c = YoloV8Config()
        assertEquals(listOf(16, 32, 64, 128, 256), listOf(64, 128, 256, 512, 1024).map(c::channels))
        assertEquals(listOf(1, 2), listOf(3, 6).map(c::repeats))
        assertEquals(8400, c.anchors)
        assertEquals(RawHead.anchors(640), c.anchors)
    }
}
