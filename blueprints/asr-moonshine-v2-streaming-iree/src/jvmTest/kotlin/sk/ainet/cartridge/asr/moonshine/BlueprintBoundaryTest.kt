package sk.ainet.cartridge.asr.moonshine

import kotlinx.serialization.json.Json
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

    @Test
    fun `no weights, compiled models, audio or built runtimes in the tree`() {
        val forbidden = setOf("safetensors", "gguf", "onnx", "vmfb", "irpa", "mlir", "so", "dylib", "dll", "bin", "zip", "wav", "pcm", "raw")
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
        val terms = Regex("""(?i)\b(intent|command|fleet|customer|tenant)\b""")
        val hits = File(dir, "src").walkTopDown().filter { it.extension == "kt" && it.path.contains("Main") }.flatMap { f ->
            f.readLines().mapIndexedNotNull { i, line -> if (terms.containsMatchIn(line)) "${f.name}:${i + 1}: ${line.trim()}" else null }
        }.toList()
        assertTrue(hits.isEmpty(), hits.joinToString("\n"))
    }

    @Test
    fun `every source is pinned, licensed, linked and freely redistributable`() {
        val bp = Json.parseToJsonElement(File(dir, "blueprint.json").readText()).jsonObject
        val sources = bp.getValue("sources").jsonArray.map { it.jsonObject }
        assertEquals(listOf("weights-en", "weights-de"), sources.map { it.getValue("name").jsonPrimitive.content })
        for (s in sources) {
            val name = s.getValue("name").jsonPrimitive.content
            assertTrue(Regex("[0-9a-f]{40}").matches(s.getValue("revision").jsonPrimitive.content), "$name: revision must be a commit id, not a branch")
            assertTrue(s.containsKey("license_url"), "$name: license_url — where the license was verified")
            assertEquals("MIT", s.getValue("license").jsonPrimitive.content, "$name: the streaming tiny checkpoints are MIT at the publisher")
            assertEquals("allowed", s.getValue("redistribution").jsonPrimitive.content)
            assertEquals(listOf("config.json", "model.safetensors", "tokenizer.json"), s.getValue("files").jsonArray.map { it.jsonObject.getValue("path").jsonPrimitive.content })
            for (f in s.getValue("files").jsonArray) assertTrue(f.jsonObject.getValue("digest").jsonPrimitive.content.startsWith("sha256:"))
        }
        val template = bp.getValue("descriptor_template").jsonObject
        for (k in listOf("performance", "quality", "target", "id", "version", "license")) assertTrue(k !in template, "descriptor_template must not contain '$k'")
        assertEquals(listOf("en", "de"), bp.getValue("flavors").jsonArray.map { it.jsonObject.getValue("id").jsonPrimitive.content })
    }

    @Test
    fun `the example profile claims nothing it has not measured`() {
        val profile = Json.parseToJsonElement(File(dir, "profiles/example.vulkan-armv7.de.json").readText()).jsonObject
        assertTrue("measurements" !in profile && "license_acceptance" !in profile)
        assertEquals(listOf("de"), profile.getValue("flavors").jsonArray.map { it.jsonPrimitive.content })
    }
}
