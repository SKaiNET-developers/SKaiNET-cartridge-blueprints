package sk.ainet.cartridge.generator.lfm25.torq

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
    fun `no weights, compiled models or built runtimes in the tree`() {
        val forbidden = setOf("safetensors", "gguf", "onnx", "vmfb", "irpa", "mlir", "so", "dylib", "dll", "bin", "zip", "npy")
        val offenders = tracked.filter { it.extension.lowercase() in forbidden }
        assertTrue(offenders.isEmpty(), "a blueprint carries no model artifacts: $offenders")
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
    fun `the sources are pinned, licensed, linked and the weights need acceptance`() {
        val bp = Json.parseToJsonElement(File(dir, "blueprint.json").readText()).jsonObject
        val sources = bp.getValue("sources").jsonArray.map { it.jsonObject }.associateBy { it.getValue("name").jsonPrimitive.content }
        assertEquals(setOf("weights", "torq-runtime-source", "iree-runtime-source"), sources.keys)
        for (s in sources.values) {
            assertTrue(Regex("[0-9a-f]{40}").matches(s.getValue("revision").jsonPrimitive.content), "revision must be a commit id, not a branch")
            assertTrue(s.containsKey("license_url"), "license_url: where the license was verified")
            for (f in s.getValue("files").jsonArray) assertTrue(f.jsonObject.getValue("digest").jsonPrimitive.content.startsWith("sha256:"))
        }
        val weights = sources.getValue("weights")
        assertEquals("LicenseRef-LFM-Open-License-1.0", weights.getValue("license").jsonPrimitive.content)
        assertEquals("restricted", weights.getValue("redistribution").jsonPrimitive.content)
        // the runtime's Torq-fork IREE code: Apache-2.0 with the LLVM exception, redistributable in the runtime library
        for (name in listOf("torq-runtime-source", "iree-runtime-source")) {
            assertEquals("Apache-2.0 WITH LLVM-exception", sources.getValue(name).getValue("license").jsonPrimitive.content)
            assertEquals("allowed", sources.getValue(name).getValue("redistribution").jsonPrimitive.content)
        }
        val runtime = bp.getValue("outputs").jsonArray.map { it.jsonObject }.single { it.getValue("role").jsonPrimitive.content == "runtime" }
        assertEquals("artifacts/runtime/liblfm2_torq_cartridge.so", runtime.getValue("path").jsonPrimitive.content)
        val template = bp.getValue("descriptor_template").jsonObject
        for (k in listOf("performance", "quality", "target", "id", "version", "license")) assertTrue(k !in template, "descriptor_template must not contain '$k'")
    }

    @Test
    fun `the example profile claims nothing it has not measured`() {
        val profile = Json.parseToJsonElement(File(dir, "profiles/example.torq-sl26xx.json").readText()).jsonObject
        assertTrue("measurements" !in profile && "license_acceptance" !in profile)
        assertEquals("torq-sl26xx", profile.getValue("target").jsonPrimitive.content)
    }
}
