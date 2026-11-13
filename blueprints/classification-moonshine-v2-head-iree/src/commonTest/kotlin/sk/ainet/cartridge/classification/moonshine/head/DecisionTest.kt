package sk.ainet.cartridge.classification.moonshine.head

import kotlin.math.abs
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

class DecisionTest {
    private val labels = LabelSet(listOf("lights_on", "lights_off", "other"), rejectLabel = "other")
    private fun decision(accept: Float, reject: Float, t: Float = 1f, score: Score = Score.ENERGY) =
        Decision(labels, Calibration(temperature = t, score = score, thresholds = Thresholds(accept, reject)))

    @Test
    fun `energy leaves the reject class out and is stable for large logits`() {
        val d = decision(0f, 0f)
        // two in-set logits of 0: E = -log(2)
        assertTrue(abs(d.score(floatArrayOf(0f, 0f, 50f)) + ln(2f)) < 1e-5f)
        // large logits must not overflow: E ≈ -1000 for one dominant class
        assertTrue(abs(d.score(floatArrayOf(1000f, 0f, 0f)) + 1000f) < 1e-2f)
        // temperature scales the logits before the sum and the result after it
        assertTrue(abs(decision(0f, 0f, t = 2f).score(floatArrayOf(4f, 4f, 0f)) - (-2f * (2f + ln(2f)))) < 1e-5f)
    }

    @Test
    fun `a confident in-set input is accepted, an unsure one undecided, a far one rejected`() {
        val d = decision(accept = -5f, reject = -1f)
        val accepted = assertIs<Outcome.Accepted>(d.decide(floatArrayOf(8f, 0f, 0f)))
        assertEquals("lights_on", accepted.label)
        assertIs<Outcome.Undecided>(d.decide(floatArrayOf(3f, 0f, 0f)))
        assertIs<Outcome.Rejected>(d.decide(floatArrayOf(0.2f, 0f, 0f)))
    }

    @Test
    fun `the reject class is never an accepted result`() {
        val d = decision(accept = 100f, reject = 100f)   // thresholds that accept everything
        assertIs<Outcome.Rejected>(d.decide(floatArrayOf(1f, 0f, 9f)))
    }

    @Test
    fun `max softmax is one minus the top probability`() {
        val d = decision(0.5f, 0.9f, score = Score.MAX_SOFTMAX)
        assertTrue(abs(d.score(floatArrayOf(0f, 0f, 0f)) - (1f - 1f / 3f)) < 1e-5f)
    }

    @Test
    fun `malformed label sets and calibrations are refused`() {
        assertFailsWith<IllegalArgumentException> { LabelSet(listOf("a", "a")) }
        assertFailsWith<IllegalArgumentException> { LabelSet(listOf("a", "b"), rejectLabel = "c") }
        assertFailsWith<IllegalArgumentException> { Thresholds(accept = 1f, reject = 0f) }
        assertFailsWith<IllegalArgumentException> { Calibration(temperature = 0f, thresholds = Thresholds(0f, 0f)) }
        assertFailsWith<IllegalArgumentException> { decision(0f, 0f).score(floatArrayOf(1f)) }
    }
}
