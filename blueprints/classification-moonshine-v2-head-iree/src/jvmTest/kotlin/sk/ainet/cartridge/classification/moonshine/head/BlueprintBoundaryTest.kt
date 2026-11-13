package sk.ainet.cartridge.classification.moonshine.head

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.cartridge.classification.moonshine.head.recipe.EncoderRows
import sk.ainet.cartridge.classification.moonshine.head.recipe.HeadExport
import sk.ainet.cartridge.classification.moonshine.head.recipe.LabelSetInput
import sk.ainet.cartridge.classification.moonshine.head.model.ClassificationHeadConfig
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The spec's blueprint MUST NOTs and ADR-018's rules for trained outputs, enforced on this module's own tree. */
class BlueprintBoundaryTest {
    private val dir = File(System.getProperty("blueprint.dir"))
    private val tracked = dir.walkTopDown().onEnter { it.name !in setOf("build", ".gradle", ".kotlin") }.filter { it.isFile }.toList()
    private val bp = Json.parseToJsonElement(File(dir, "blueprint.json").readText()).jsonObject
    private fun list(name: String): List<JsonObject> = bp.getValue(name).jsonArray.map { it.jsonObject }
    private fun JsonObject.strings(name: String): List<String> = this[name]?.jsonArray?.map { it.jsonPrimitive.content } ?: emptyList()

    @Test
    fun `no weights, compiled models, recordings, datasets or built runtimes in the tree`() {
        val forbidden = setOf("pt", "pth", "safetensors", "gguf", "onnx", "vmfb", "irpa", "mlir", "so", "dylib", "dll", "bin", "zip", "wav", "flac", "mp3", "parquet", "npy")
        val offenders = tracked.filter { it.extension.lowercase() in forbidden }
        assertTrue(offenders.isEmpty(), "a blueprint carries no model artifacts and no data: $offenders")
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
    fun `the encoders are pinned, and no dataset is a source`() {
        val sources = list("sources")
        assertEquals(listOf("encoder-de", "encoder-en"), sources.map { it.getValue("name").jsonPrimitive.content })
        for (s in sources) {
            val revision = s.getValue("revision").jsonPrimitive.content
            assertTrue(Regex("[0-9a-f]{40}").matches(revision), "revision must be a commit id, not a branch")
            assertTrue(s.getValue("license_url").jsonPrimitive.content.contains(revision), "license_url pinned to the revision")
            assertEquals("MIT", s.getValue("license").jsonPrimitive.content)
            val files = s.getValue("files").jsonArray.map { it.jsonObject }
            assertEquals(listOf("config.json", "model.safetensors"), files.map { it.getValue("path").jsonPrimitive.content })
            for (f in files) assertTrue(Regex("sha256:[0-9a-f]{64}").matches(f.getValue("digest").jsonPrimitive.content))
        }
        // The data is the materializing party's: an input with their license, never something the blueprint ships or pins.
        assertEquals(listOf("label-set", "dataset"), list("inputs").map { it.getValue("name").jsonPrimitive.content })
        for (input in list("inputs")) assertTrue(File(dir, input.getValue("schema").jsonPrimitive.content).isFile)
    }

    @Test
    fun `train and calibrate steps declare their data and their seed, and trained outputs descend from the data`() {
        val fitted = list("steps").filter { it.getValue("kind").jsonPrimitive.content in setOf("train", "calibrate") }
        assertEquals(listOf("train", "calibrate"), fitted.map { it.getValue("id").jsonPrimitive.content })
        for (step in fitted) {
            assertTrue("dataset" in step.strings("uses"), "${step["id"]}: uses the dataset")
            assertTrue(step.getValue("params").jsonObject.getValue("seed").jsonPrimitive.int >= 0, "${step["id"]}: params.seed")
        }
        val trained = list("outputs").filter { it.getValue("role").jsonPrimitive.content in setOf("model", "weights", "calibration") }
        assertEquals(6, trained.size)
        for (o in trained) {
            val flavor = o.getValue("flavor").jsonPrimitive.content
            assertEquals(listOf("encoder-$flavor", "dataset", "label-set"), o.strings("derived_from"), o.getValue("path").jsonPrimitive.content)
        }
        assertTrue(list("verification").any { it.getValue("kind").jsonPrimitive.content == "task-metric" && "tolerance" in it },
            "trained weights are not byte-reproducible: a task-metric with a tolerance stands in (ADR-018)")
    }

    @Test
    fun `the descriptor template states the contract the head is built to`() {
        val template = bp.getValue("descriptor_template").jsonObject
        for (k in listOf("performance", "quality", "target", "id", "version", "license")) assertTrue(k !in template, "descriptor_template must not contain '$k'")
        assertEquals("classification", template.getValue("task").jsonPrimitive.content)
        val attrs = template.getValue("attributes").jsonObject
        assertEquals("classification/v1", attrs.getValue("task_profile").jsonPrimitive.content)
        val outputs = list("outputs").map { it.getValue("path").jsonPrimitive.content }
        assertTrue(PackDir.LABEL_SET in outputs && PackDir.CALIBRATION in outputs, "classification/v1 fixes where the label set and the calibration are")

        val config = ClassificationHeadConfig(classes = 2)
        val io = template.getValue("io").jsonObject
        assertEquals("batch", io.getValue("mode").jsonPrimitive.content)
        assertEquals(listOf(config.maxRows, config.rowWidth), io.getValue("input").jsonObject.getValue("shape").jsonArray.map { it.jsonPrimitive.int })
        val features = attrs.getValue("features").jsonObject
        assertEquals(config.dim, features.getValue("dim").jsonPrimitive.int)
        assertEquals(config.maxRows, features.getValue("max_rows").jsonPrimitive.int)
        assertEquals(EncoderRows.SAMPLE_RATE / EncoderRows.FRAME_SAMPLES, features.getValue("rows_per_second").jsonPrimitive.int)
        val window = features.getValue("window").jsonObject.mapValues { it.value.jsonPrimitive.int }
        assertEquals(mapOf("chunk" to EncoderRows.CHUNK, "left" to EncoderRows.LEFT, "lookahead" to EncoderRows.LOOKAHEAD, "hop" to EncoderRows.HOP), window)
        assertEquals(HeadExport.FUNCTION, attrs.getValue("runtime_hint").jsonObject.getValue("function").jsonPrimitive.content)
        assertEquals(config.maxRows, list("steps").single { it.getValue("id").jsonPrimitive.content == "features" }.getValue("params").jsonObject.getValue("max_rows").jsonPrimitive.int)
    }

    @Test
    fun `every flavor is one encoder, and names it in the feature contract`() {
        for (flavor in list("flavors")) {
            val id = flavor.getValue("id").jsonPrimitive.content
            assertEquals(listOf("encoder-$id"), flavor.strings("sources"))
            val source = list("sources").single { it.getValue("name").jsonPrimitive.content == "encoder-$id" }
            val features = flavor.getValue("descriptor_overrides").jsonObject.getValue("attributes").jsonObject.getValue("features").jsonObject
            assertEquals(source.getValue("revision"), features.getValue("revision"))
            assertEquals(source.getValue("uri").jsonPrimitive.content.removePrefix("hf://"), features.getValue("encoder").jsonPrimitive.content)
        }
    }

    @Test
    fun `every target names backend, IREE target and device`() {
        for (t in list("targets")) {
            val params = t.getValue("params").jsonObject
            for (k in listOf("backend", "iree_target", "device")) assertTrue(k in params, "${t["id"]}: params.$k")
            val accelerated = t.getValue("target").jsonObject.containsKey("accelerator")
            assertEquals(accelerated, t.getValue("requirements").jsonObject.getValue("needs_accelerator").jsonPrimitive.content.toBoolean())
        }
    }

    @Test
    fun `the example claims nothing it has not measured, and its label set is a valid input`() {
        val profile = Json.parseToJsonElement(File(dir, "profiles/example.vulkan-armv7.de.json").readText()).jsonObject
        assertTrue("measurements" !in profile && "license_acceptance" !in profile)
        val labels = LabelSetInput.read(File(dir, "samples/label-set.example.json"))
        assertEquals("other", labels.labelSet.rejectLabel)
        assertTrue(labels.heldOut.isNotEmpty())
    }
}
