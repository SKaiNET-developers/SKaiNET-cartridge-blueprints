package sk.ainet.cartridge.generator.lfm25.torq

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DecodeStepTest {
    // A small step in the exporter's io.txt form: emitter order, controls interleaved with state.
    private val io = """
        in 0 write_mask [1, 4, 1]
        in 1 embedded [1, 2]
        in 2 history_0 [2, 3]
        in 3 cos [1, 2]
        in 4 sin [1, 2]
        in 5 key_1 [1, 4, 2]
        in 6 value_1 [1, 4, 2]
        in 7 mask [1, 5]
        out 0 history_0 [2, 3]
        out 1 key_1 [1, 4, 2]
        out 2 value_1 [1, 4, 2]
        out 3 logits [1, 3]
    """.trimIndent()

    private val ports = DecodeStepPorts.parse(io)

    @Test
    fun portsKeepTheArgumentOrderAndFindTheState() {
        assertEquals(listOf("write_mask", "embedded", "history_0", "cos", "sin", "key_1", "value_1", "mask"), ports.inputs.map { it.role })
        assertEquals(listOf("history_0", "key_1", "value_1"), ports.state.map { it.role })
        assertEquals(2, ports.width); assertEquals(2, ports.headDim); assertEquals(4, ports.capacity); assertEquals(3, ports.vocabulary)
        assertFailsWith<IllegalArgumentException> { DecodeStepPorts.parse(io.replace("out 0 history_0 [2, 3]", "out 0 history_0 [2, 4]")) }
        assertFailsWith<IllegalArgumentException> { DecodeStepPorts.parse(io.lines().filterNot { "mask [1, 5]" in it }.joinToString("\n")) }
    }

    @Test
    fun controlsMaskTheUnfilledSlotsAndWriteOneSlot() {
        val c = StepControls(headDim = 4, ropeTheta = 1e6, capacity = 4).forPosition(2)
        assertContentEquals(floatArrayOf(0f, 0f, StepControls.MASKED, StepControls.MASKED, 0f), c.mask)
        assertContentEquals(floatArrayOf(0f, 0f, 1f, 0f), c.writeMask)
        // split-half RoPE: the same angle in both halves; lane 0 turns by the position itself
        assertTrue(abs(c.cos[0] - kotlin.math.cos(2.0).toFloat()) < 1e-6f && c.cos[0] == c.cos[2] && c.sin[1] == c.sin[3])
        assertFailsWith<IllegalArgumentException> { StepControls(4, 1e6, 4).forPosition(4) }
    }

    @Test
    fun embeddingRowsDecodeLittleEndianBf16() {
        // 1.0 = 0x3F80, -2.0 = 0xC000, 0.5 = 0x3F00, 3.0 = 0x4040
        val bytes = byteArrayOf(0x80.toByte(), 0x3F, 0x00, 0xC0.toByte(), 0x00, 0x3F, 0x40, 0x40)
        val rows = EmbeddingRows(vocabulary = 2, width = 2, bf16 = bytes)
        assertContentEquals(floatArrayOf(1f, -2f), rows.row(0))
        assertContentEquals(floatArrayOf(0.5f, 3f), rows.row(1))
        assertFailsWith<IllegalArgumentException> { EmbeddingRows(2, 3, bytes) }
    }

    /** A step that counts in its state and answers token (position + 1) mod 3, recording what it was fed. */
    private class CountingRunner : DecodeStepRunner {
        val positions = ArrayList<Int>()
        override fun run(inputs: Map<String, FloatArray>): Map<String, FloatArray> {
            val position = inputs.getValue("write_mask").indexOfFirst { it == 1f }
            positions += position
            val history = inputs.getValue("history_0")
            assertEquals(position.toFloat(), history[0], "state threads from step to step")
            val logits = FloatArray(3).also { it[(position + 1) % 3] = 1f }
            return mapOf(
                "history_0" to history.copyOf().also { it[0] += 1f },
                "key_1" to inputs.getValue("key_1"), "value_1" to inputs.getValue("value_1"), "logits" to logits,
            )
        }
    }

    private fun decoder(runner: DecodeStepRunner) =
        GreedyDecoder(ports, StepControls(2, 1e6, 4), EmbeddingRows(3, 2, ByteArray(12)), runner)

    @Test
    fun greedyDecodeFeedsThePromptThenItsOwnTokens() {
        val runner = CountingRunner()
        val g = decoder(runner).generate(intArrayOf(0), maxNewTokens = 2, stopTokens = emptySet())
        assertContentEquals(intArrayOf(1, 2), g.tokens)
        assertEquals(FinishReason.LENGTH, g.finishReason)
        assertEquals(listOf(0, 1), runner.positions)
    }

    @Test
    fun greedyDecodeStopsAtAStopTokenAndAtTheCapacity() {
        assertEquals(FinishReason.STOP, decoder(CountingRunner()).generate(intArrayOf(0), 5, stopTokens = setOf(2)).finishReason)
        val full = decoder(CountingRunner()).generate(intArrayOf(0, 1), maxNewTokens = 10, stopTokens = emptySet())
        assertEquals(FinishReason.CAPACITY, full.finishReason)
        assertEquals(3, full.tokens.size)
        val d = decoder(CountingRunner())
        d.generate(intArrayOf(0), 1, emptySet())
        d.reset()
        assertEquals(0, d.position)
    }
}
