package sk.ainet.cartridge.nlu.functiongemma

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
    private val tracked = dir.walkTopDown()
        .onEnter { it.name !in setOf("build", ".gradle", ".kotlin") }
        .filter { it.isFile }.toList()

    @Test
    fun `no weights, compiled models or built runtimes in the tree`() {
        val forbidden = setOf("safetensors", "gguf", "onnx", "ort", "vmfb", "irpa", "mlir", "so", "dylib", "dll", "bin", "zip")
        val offenders = tracked.filter { it.extension.lowercase() in forbidden }
        assertTrue(offenders.isEmpty(), "a blueprint carries no model artifacts: $offenders")
    }

    @Test
    fun `the cartridge code contains no catalog of its own`() {
        // Which functions exist is the host's knowledge. The only catalog in this repository is the toy sample.
        val main = File(dir, "src").walkTopDown().filter { it.extension == "kt" && it.path.contains("Main") }.toList()
        val offenders = main.filter { f -> Regex("""ToolFunction\(\s*"|"functions"\s*:""").containsMatchIn(f.readText()) }
        assertTrue(offenders.isEmpty(), "main sources must not define functions: $offenders")
        assertEquals(listOf("toy-catalog.json", "toy-golden.json"), File(dir, "samples").list()!!.sorted())
    }

    @Test
    fun `no application, device or organization vocabulary`() {
        // ("session" is absent from the list on purpose: it names the runtime's KV session.)
        val terms = Regex("""(?i)\b(intent|command[-_ ]?assembler|fleet|customer|tenant)\b""")
        val hits = File(dir, "src").walkTopDown().filter { it.extension == "kt" && it.path.contains("Main") }.flatMap { f ->
            f.readLines().mapIndexedNotNull { i, line -> if (terms.containsMatchIn(line)) "${f.name}:${i + 1}: ${line.trim()}" else null }
        }.toList()
        assertTrue(hits.isEmpty(), hits.joinToString("\n"))
    }

    @Test
    fun `every source is pinned, licensed and linked`() {
        val bp = Json.parseToJsonElement(File(dir, "blueprint.json").readText()).jsonObject
        for (s in bp.getValue("sources").jsonArray.map { it.jsonObject }) {
            val name = s.getValue("name").jsonPrimitive.content
            assertTrue(Regex("[0-9a-f]{40}").matches(s.getValue("revision").jsonPrimitive.content), "$name: revision must be a commit id, not a branch")
            assertTrue(s.containsKey("license_url"), "$name: license_url — where the license was verified")
            for (f in s.getValue("files").jsonArray) assertTrue(f.jsonObject.getValue("digest").jsonPrimitive.content.startsWith("sha256:"))
        }
        val template = bp.getValue("descriptor_template").jsonObject
        for (k in listOf("performance", "quality", "target", "id", "version", "license")) assertTrue(k !in template, "descriptor_template must not contain '$k'")
    }

    @Test
    fun `the example profile accepts no license and claims no measurement`() {
        for (p in File(dir, "profiles").listFiles { f -> f.extension == "json" }!!) {
            val profile = Json.parseToJsonElement(p.readText()).jsonObject
            assertTrue("license_acceptance" !in profile, "${p.name}: acceptance is recorded by whoever builds, never shipped")
            assertTrue("measurements" !in profile, "${p.name}: measurements come from a device, never from the repository")
        }
    }
}
