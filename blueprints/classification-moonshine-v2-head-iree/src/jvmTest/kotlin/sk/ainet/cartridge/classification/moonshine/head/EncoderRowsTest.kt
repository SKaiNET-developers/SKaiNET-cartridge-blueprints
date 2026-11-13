package sk.ainet.cartridge.classification.moonshine.head

import sk.ainet.cartridge.classification.moonshine.head.recipe.EncoderRows
import java.io.File
import kotlin.math.PI
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Needs the publisher's checkpoint, which a blueprint never carries: set MOONSHINE_MODEL_DIR to a
 * `moonshine-streaming-tiny-de` snapshot (config.json, model.safetensors). Skipped — loudly — without it.
 */
class EncoderRowsTest {
    private val modelDir = System.getenv("MOONSHINE_MODEL_DIR")?.let(::File)

    private fun tone(seconds: Double): FloatArray =
        FloatArray((seconds * EncoderRows.SAMPLE_RATE).toInt()) { i ->
            (0.3 * sin(2 * PI * 220 * i / EncoderRows.SAMPLE_RATE) + 0.1 * sin(2 * PI * 1870 * i / EncoderRows.SAMPLE_RATE)).toFloat()
        }

    @Test
    fun `rows come at 50 per second, once each, capped at the maximum`() {
        if (modelDir == null) { println("SKIPPED EncoderRowsTest: set MOONSHINE_MODEL_DIR to a moonshine-streaming-tiny-de snapshot"); return }
        EncoderRows(modelDir).use { encoder ->
            assertEquals(320, encoder.dim)
            val started = System.nanoTime()
            val (rows, n) = encoder.rows(tone(3.0), maxRows = 256)
            val ms = (System.nanoTime() - started) / 1_000_000
            println("EncoderRows: 3.0 s of audio -> $n rows in $ms ms")
            assertEquals(150, n)
            assertTrue(rows.take(n * 320).all { it.isFinite() })
            assertTrue(rows.drop(n * 320).all { it == 0f }, "rows past the end stay zero")

            val (_, short) = encoder.rows(tone(0.5), maxRows = 256)
            assertEquals(25, short)
            val (_, capped) = encoder.rows(tone(7.0), maxRows = 256)
            assertEquals(256, capped)

            // A row that an earlier window finalized is not changed by audio that comes after its lookahead.
            val (a, _) = encoder.rows(tone(3.0), maxRows = 256)
            val (b, _) = encoder.rows(tone(4.0), maxRows = 256)
            for (i in 0 until 60 * 320) assertEquals(a[i], b[i], 1e-4f)
        }
    }
}
