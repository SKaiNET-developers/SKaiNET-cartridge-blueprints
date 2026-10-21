package sk.ainet.cartridge.detection.yolov8

import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PackDirTest {
    private fun pack(accelerator: Boolean, attributes: String, withFiles: Boolean = true): File {
        val root = Files.createTempDirectory("yolov8-pack").toFile()
        val target = if (accelerator) """{"hardware":"IREE-arm64-v8a","abi":"arm64-v8a","accelerator":"Vulkan GPU (valhall4)"}""" else """{"hardware":"IREE-arm64-v8a","abi":"arm64-v8a"}"""
        File(root, "descriptor.json").writeText("""{"id":"detection-yolov8-iree-cpu-arm64","version":"0.1.0","target":$target,"attributes":$attributes}""")
        if (withFiles) {
            File(root, "artifacts/model/yolov8n.vmfb").apply { parentFile.mkdirs() }.writeBytes(byteArrayOf(1))
            File(root, "artifacts/weights/params.irpa").apply { parentFile.mkdirs() }.writeBytes(byteArrayOf(2))
        }
        return root
    }

    @Test
    fun readsSettingsAndLocatesTheModel() {
        val p = PackDir(pack(accelerator = true, attributes = """{"label_set":["person","bicycle"],"input_resolution":320,"conf_threshold":0.5,"iou_threshold":0.6,"max_detections":10}""").path)
        assertEquals("detection-yolov8-iree-cpu-arm64", p.id)
        assertEquals("0.1.0", p.version)
        assertTrue(p.targetsAccelerator)
        assertEquals(DetectionSettings(listOf("person", "bicycle"), 320, 0.5f, 0.6f, 10), p.settings)
        assertTrue(p.model().vmfb.toString().endsWith("artifacts/model/yolov8n.vmfb"))
        assertTrue(p.model().paramsIrpa.toString().endsWith("artifacts/weights/params.irpa"))
    }

    @Test
    fun missingAttributesFallBackToTheBlueprintDefaults() {
        val p = PackDir(pack(accelerator = false, attributes = "{}").path)
        assertTrue(!p.targetsAccelerator)
        assertEquals(DetectionSettings(), p.settings)
    }

    @Test
    fun aMissingFileIsReportedByName() {
        val p = PackDir(pack(accelerator = false, attributes = "{}", withFiles = false).path)
        val e = assertFailsWith<IllegalArgumentException> { p.model() }
        assertTrue(e.message!!.contains("artifacts/model/yolov8n.vmfb"))
        assertFailsWith<IllegalArgumentException> { PackDir(Files.createTempDirectory("empty").toString()) }
    }
}
