package sk.ainet.cartridge.classification.moonshine.head

import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PackDirTest {
    private fun pack(): File {
        val dir = Files.createTempDirectory("pack").toFile().apply { deleteOnExit() }
        File(dir, "descriptor.json").writeText(
            """{"id":"head-test","target":{"hardware":"x","abi":"arm64-v8a","accelerator":"Vulkan GPU"},
               "attributes":{"task_profile":"classification/v1","features":{"encoder":"org/encoder","revision":"abc","dim":2,"max_rows":3},
                             "runtime_hint":{"function":"classify"}}}""",
        )
        File(dir, "artifacts/other").mkdirs()
        File(dir, "artifacts/calibration").mkdirs()
        File(dir, PackDir.LABEL_SET).writeText("""{"labels":["a","b","other"],"reject_label":"other"}""")
        File(dir, PackDir.CALIBRATION).writeText("""{"temperature":2.0,"score":"energy","thresholds":{"accept":-3.0,"reject":-1.0},"target_risk":0.01}""")
        return dir
    }

    @Test
    fun `a pack gives its feature contract, its labels and its decision`() {
        val pack = PackDir(pack().path)
        assertEquals("head-test", pack.id)
        assertTrue(pack.targetsAccelerator)
        assertEquals("classify", pack.function)
        assertEquals(PackDir.Features("org/encoder", "abc", dim = 2, maxRows = 3), pack.features)
        assertEquals(listOf("a", "b", "other"), pack.labels.labels)
        assertEquals(2f, pack.calibration.temperature)

        // Class a wins clearly: energy = -T * log(exp(10/2) + exp(0)) is about -10, below accept.
        val accepted = pack.decision().decide(floatArrayOf(10f, 0f, 0f))
        assertEquals("a", (accepted as Outcome.Accepted).label)
        // The reject class on top is a refusal whatever the score.
        assertTrue(pack.decision().decide(floatArrayOf(0f, 0f, 10f)) is Outcome.Rejected)
    }

    @Test
    fun `the input is the rows with a mask column, padded and cut to max_rows`() {
        val features = PackDir(pack().path).features
        assertEquals(9, features.inputSize)
        assertEquals(listOf(1f, 2f, 1f, 3f, 4f, 1f, 0f, 0f, 0f), features.input(floatArrayOf(1f, 2f, 3f, 4f), count = 2).toList())
        val cut = features.input(FloatArray(8) { it + 1f }, count = 4)
        assertEquals(listOf(1f, 2f, 1f, 3f, 4f, 1f, 5f, 6f, 1f), cut.toList())
        assertFailsWith<IllegalArgumentException> { features.input(floatArrayOf(1f), count = 2) }
    }

    @Test
    fun `a directory without a descriptor is not a pack, and a missing artifact is named`() {
        assertFailsWith<IllegalArgumentException> { PackDir(Files.createTempDirectory("empty").toString()) }
        val error = assertFailsWith<IllegalArgumentException> { PackDir(pack().path).module }
        assertTrue("artifacts/model/head.vmfb" in error.message!!)
    }
}
