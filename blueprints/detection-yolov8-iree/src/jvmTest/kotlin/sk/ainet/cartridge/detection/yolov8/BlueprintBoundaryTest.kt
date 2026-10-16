package sk.ainet.cartridge.detection.yolov8

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The spec's blueprint MUST NOTs, enforced on this module's own tree. */
class BlueprintBoundaryTest {
    private val dir = File(System.getProperty("blueprint.dir"))
    private val tracked = dir.walkTopDown().onEnter { it.name !in setOf("build", ".gradle", ".kotlin") }.filter { it.isFile }.toList()
    private val bp = Json.parseToJsonElement(File(dir, "blueprint.json").readText()).jsonObject

    @Test
    fun `no weights, compiled models, images or built runtimes in the tree`() {
        val forbidden = setOf("pt", "pth", "safetensors", "gguf", "onnx", "vmfb", "irpa", "mlir", "so", "dylib", "dll", "bin", "zip", "jpg", "jpeg", "png")
        val offenders = tracked.filter { it.extension.lowercase() in forbidden }
        assertTrue(offenders.isEmpty(), "a blueprint carries no model artifacts: $offenders")
    }

    @Test
    fun `no C sources — the native runtime is a released artifact`() {
        val offenders = tracked.filter { it.extension.lowercase() in setOf("c", "cc", "cpp", "h", "cmake") || it.name == "CMakeLists.txt" }
        assertTrue(offenders.isEmpty(), "the runtime comes from SKaiNET-transformers, not from here: $offenders")
    }

    @Test
    fun `no application, device or organization vocabulary`() {
        val terms = Regex("""(?i)\b(intent|command|fleet|customer|tenant|router)\b""")
        val hits = File(dir, "src").walkTopDown().filter { it.extension == "kt" && it.path.contains("Main") }.flatMap { f ->
            f.readLines().mapIndexedNotNull { i, line -> if (terms.containsMatchIn(line)) "${f.name}:${i + 1}: ${line.trim()}" else null }
        }.toList()
        assertTrue(hits.isEmpty(), hits.joinToString("\n"))
    }

    @Test
    fun `the weights are pinned, linked, and flagged for their copyleft license`() {
        val sources = bp.getValue("sources").jsonArray.map { it.jsonObject }
        assertEquals(listOf("weights"), sources.map { it.getValue("name").jsonPrimitive.content })
        val s = sources.single()
        assertTrue(Regex("[0-9a-f]{40}").matches(s.getValue("revision").jsonPrimitive.content), "revision must be a commit id, not a branch")
        assertTrue(s.getValue("license_url").jsonPrimitive.content.contains(s.getValue("revision").jsonPrimitive.content), "license_url pinned to the revision")
        assertEquals("AGPL-3.0-only", s.getValue("license").jsonPrimitive.content)
        assertEquals("restricted", s.getValue("redistribution").jsonPrimitive.content, "AGPL obligations travel with every cartridge")
        val files = s.getValue("files").jsonArray.map { it.jsonObject }
        assertEquals(listOf("yolov8n.pt"), files.map { it.getValue("path").jsonPrimitive.content })
        for (f in files) assertTrue(Regex("sha256:[0-9a-f]{64}").matches(f.getValue("digest").jsonPrimitive.content))
    }

    @Test
    fun `every container tool is pinned by digest`() {
        for ((name, tool) in bp.getValue("toolchain").jsonObject) {
            val image = tool.jsonObject["image"]?.jsonPrimitive?.content ?: continue
            if (name == "iree-tools") continue  // the IREE tools image is built locally from a pinned recipe (skainet/iree-compiler)
            assertTrue(image.contains("@sha256:"), "$name: $image must be pinned by digest")
        }
    }

    @Test
    fun `the descriptor template states the io contract the adapter decodes`() {
        val template = bp.getValue("descriptor_template").jsonObject
        for (k in listOf("performance", "quality", "target", "id", "version", "license")) assertTrue(k !in template, "descriptor_template must not contain '$k'")
        val attrs = template.getValue("attributes").jsonObject
        val labels = attrs.getValue("label_set").jsonArray
        val resolution = attrs.getValue("input_resolution").jsonPrimitive.int
        val io = template.getValue("io").jsonObject
        assertEquals("batch", io.getValue("mode").jsonPrimitive.content)
        assertEquals(listOf(1, 3, resolution, resolution), io.getValue("input").jsonObject.getValue("shape").jsonArray.map { it.jsonPrimitive.int })
        assertEquals(listOf(1, 4 + labels.size, RawHead.anchors(resolution)), io.getValue("output").jsonObject.getValue("shape").jsonArray.map { it.jsonPrimitive.int })
    }

    @Test
    fun `every target names backend, IREE target and device`() {
        for (t in bp.getValue("targets").jsonArray.map { it.jsonObject }) {
            val params = t.getValue("params").jsonObject
            for (k in listOf("backend", "iree_target", "device")) assertTrue(k in params, "${t["id"]}: params.$k")
            val accelerated = t.getValue("target").jsonObject.containsKey("accelerator")
            assertEquals(accelerated, t.getValue("requirements").jsonObject.getValue("needs_accelerator").jsonPrimitive.content.toBoolean())
        }
    }

    @Test
    fun `the example profile claims nothing it has not measured or accepted`() {
        val profile = Json.parseToJsonElement(File(dir, "profiles/example.cpu-arm64.json").readText()).jsonObject
        assertTrue("measurements" !in profile && "license_acceptance" !in profile)
    }
}
