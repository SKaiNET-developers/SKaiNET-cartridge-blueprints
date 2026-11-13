package sk.ainet.cartridge.classification.moonshine.head

import sk.ainet.cartridge.classification.moonshine.head.recipe.Example
import sk.ainet.cartridge.classification.moonshine.head.recipe.FeatureStore
import sk.ainet.cartridge.classification.moonshine.head.recipe.LabelSetInput
import sk.ainet.cartridge.classification.moonshine.head.recipe.Metrics
import sk.ainet.cartridge.classification.moonshine.head.recipe.Npy
import sk.ainet.cartridge.classification.moonshine.head.recipe.Wav
import sk.ainet.cartridge.classification.moonshine.head.recipe.examples
import java.io.File
import java.nio.file.Files
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RecipeToolsTest {
    private val dir: File = Files.createTempDirectory("head-tools").toFile().apply { deleteOnExit() }

    @Test
    fun `the temperature fit recovers the temperature the labels were drawn with`() {
        val rng = Random(3)
        val truth = 2.5f
        val logits = List(4000) { FloatArray(5) { (rng.nextFloat() - 0.5f) * 12f } }
        val targets = IntArray(logits.size) { i ->
            // A label drawn from softmax(z / truth): the fit should find `truth`.
            val p = logits[i].map { Math.exp((it / truth).toDouble()) }
            var r = rng.nextDouble() * p.sum()
            var c = 0
            while (c < p.size - 1 && r > p[c]) { r -= p[c]; c++ }
            c
        }
        val fitted = Metrics.fitTemperature(logits, targets)
        assertTrue(abs(fitted - truth) < 0.2f, "fitted $fitted, drawn with $truth")
    }

    @Test
    fun `the accept threshold is the loosest one that keeps the risk`() {
        //                   right  right  wrong  right  right  wrong  (not acceptable)
        val scores = floatArrayOf(-9f, -8f, -7f, -6f, -5f, -4f, -10f)
        val correct = booleanArrayOf(true, true, false, true, true, false, true)
        val acceptable = booleanArrayOf(true, true, true, true, true, true, false)
        // No errors allowed: only the first two.
        assertEquals(-8f, Metrics.acceptThreshold(scores, acceptable, correct, 0f))
        // One in five: up to -5 (1 wrong of 5).
        assertEquals(-5f, Metrics.acceptThreshold(scores, acceptable, correct, 0.2f))
        // The most confident acceptable input is wrong and no prefix recovers: nothing is accepted.
        val none = Metrics.acceptThreshold(floatArrayOf(-3f, -2f), booleanArrayOf(true, true), booleanArrayOf(false, false), 0.1f)
        assertTrue(none < -3f)
    }

    @Test
    fun `separation and calibration metrics on known cases`() {
        assertEquals(1.0, Metrics.auroc(floatArrayOf(-3f, -2f), floatArrayOf(1f, 2f)))
        assertEquals(0.0, Metrics.auroc(floatArrayOf(1f, 2f), floatArrayOf(-3f, -2f)))
        assertEquals(0.5, Metrics.auroc(floatArrayOf(1f), floatArrayOf(1f)))
        assertEquals(0.5, Metrics.fprAtTpr(floatArrayOf(0f, 1f, 2f, 3f), floatArrayOf(0.5f, 9f), tpr = 0.5f))
        // Two classes, one error each way: F1 is 0.5 for both.
        assertEquals(0.5, Metrics.macroF1(intArrayOf(0, 1, 0, 1), intArrayOf(0, 0, 1, 1), 2))
        // Certain and right everywhere: no calibration error.
        assertEquals(0.0, Metrics.ece(listOf(floatArrayOf(50f, 0f), floatArrayOf(0f, 50f)), intArrayOf(0, 1), 1f), 1e-9)
    }

    @Test
    fun `features survive their shards and get their targets from the label set`() {
        val store = FeatureStore(File(dir, "features"))
        val header = FeatureStore.Header("sha256:m", "sha256:e", dim = 2, maxRows = 4, entries = 4)
        store.open(header)
        fun item(label: String, split: String, rows: Int) = FeatureStore.Item("$label.wav", label, split, rows)
        store.write(0, listOf(item("a", "train", 2), item("b", "test", 1), item("zzz", "test", 3), item("held", "test", 1)),
            listOf(floatArrayOf(1f, 2f, 3f, 4f), floatArrayOf(5f, 6f), floatArrayOf(7f, 8f, 9f, 10f, 11f, 12f), floatArrayOf(13f, 14f)), dim = 2)
        store.open(header)   // the same header keeps what is there
        assertTrue(store.has(0))

        val labels = LabelSetInput(listOf("a", "b", "other"), rejectLabel = "other", heldOut = listOf("held"))
        val test = store.examples("test", labels)
        assertEquals(listOf(1, 2, 2), test.map { it.target })
        assertEquals(listOf(false, false, true), test.map { it.unseen })
        assertContentEquals(floatArrayOf(7f, 8f, 9f, 10f, 11f, 12f), test[1].rows)
        assertEquals(listOf(0), store.examples("train", labels).map { it.target })

        // Without a reject class, an unlisted label is out of set.
        val bare = LabelSetInput(listOf("a", "b"))
        assertEquals(listOf(1, Example.OUT_OF_SET, Example.OUT_OF_SET), store.examples("test", bare).map { it.target })

        store.open(header.copy(manifest = "sha256:other"))   // other data: start over
        assertTrue(!store.has(0))
    }

    @Test
    fun `wav files round trip and a wrong rate is refused by name`() {
        val pcm = FloatArray(1600) { sin(2 * PI * 440 * it / 16000).toFloat() * 0.5f }
        val file = File(dir, "tone.wav")
        Wav.write(file, pcm, 16_000)
        val back = Wav.read(file)
        assertEquals(pcm.size, back.size)
        for (i in pcm.indices) assertEquals(pcm[i], back[i], 1e-4f)

        val fast = File(dir, "fast.wav")
        Wav.write(fast, pcm, 48_000)
        val error = assertFailsWith<IllegalArgumentException> { Wav.read(fast) }
        assertTrue("48000 Hz" in error.message!! && "fast.wav" in error.message!!, error.message)
    }

    @Test
    fun `npy files round trip`() {
        val file = File(dir, "x.npy")
        val values = FloatArray(24) { it * 0.5f }
        Npy.write(file, intArrayOf(1, 4, 6), values)
        assertEquals(0, (file.length().toInt() - 24 * 4) % 64, "the header is padded to 64 bytes")
        assertContentEquals(values, Npy.read(file))
    }
}
