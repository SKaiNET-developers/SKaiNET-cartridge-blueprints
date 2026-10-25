package sk.ainet.cartridge.generator.llama32

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import sk.ainet.cartridge.profiles.llm.LlmEvent
import sk.ainet.cartridge.profiles.llm.LlmFinishReason
import sk.ainet.cartridge.profiles.llm.LlmMessage
import sk.ainet.cartridge.profiles.llm.LlmRole
import sk.ainet.cartridge.profiles.llm.LlmSessionConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * End-to-end over the real pinned GGUF (never in git): skips loudly when the checkpoint is not
 * on disk (`~/.cache/skainet/models/unsloth_Llama-3.2-1B-Instruct-GGUF` or LLAMA32_MODEL_DIR).
 * Greedy decode (temperature 0) is deterministic, so two runs must agree token for token —
 * the same determinism claim the SmolLM2 goldens make, pinned here against the model itself
 * until a committed golden transcript joins it.
 */
class Llama32GenerationSmokeTest {

    private fun modelAvailable(): Boolean {
        val dir = Llama32GeneratorCartridge.defaultModelDir() ?: return false
        return fileSize("$dir/${Llama32GeneratorCartridge.MODEL_FILE}") ==
            Llama32GeneratorCartridge.MODEL_BYTES
    }

    // runBlocking, not runTest: runTest's timeout counts wall-clock time, so a laptop sleeping
    // mid-decode fails the test spuriously (seen once: 10m timeout, 8h18m of wall clock).
    @Test
    fun greedyDecodeIsDeterministicAndStops() = runBlocking {
        if (!modelAvailable()) {
            println(
                "SKIPPED: pinned checkpoint not found — put ${Llama32GeneratorCartridge.MODEL_FILE} in " +
                    "${Llama32GeneratorCartridge.defaultModelDir()} or set LLAMA32_MODEL_DIR",
            )
            return@runBlocking
        }
        val engine = Llama32GeneratorCartridge.open()
        try {
            val messages = listOf(
                LlmMessage(
                    LlmRole.SYSTEM,
                    "You are a translator. Translate the user's text to German. Reply with the translation only.",
                ),
                LlmMessage(LlmRole.USER, "The weather is beautiful today."),
            )
            val config = LlmSessionConfig(temperature = 0f, maxOutputTokens = 64)

            val first = decode(engine, messages, config)
            val second = decode(engine, messages, config)

            assertEquals(LlmFinishReason.STOP, first.finishReason, "greedy decode must hit a stop token")
            assertTrue(first.text.isNotBlank(), "generated text must not be empty")
            assertEquals(first.tokens, second.tokens, "greedy decode must be deterministic")
            println("greedy output: ${first.text}")
        } finally {
            engine.close()
        }
    }

    private suspend fun decode(
        engine: Llama32Engine,
        messages: List<LlmMessage>,
        config: LlmSessionConfig,
    ): LlmEvent.Final {
        val session = engine.session(config)
        session.prompt(messages)
        val events = session.events.toList()
        session.close()
        assertIs<LlmEvent.Ready>(events.first())
        val terminal = events.last()
        assertIs<LlmEvent.Final>(terminal, "terminal event must be Final, got $terminal")
        // The deltas must concatenate to the final text — the llm/v1 ordering contract.
        val joined = events.filterIsInstance<LlmEvent.Delta>().joinToString("") { it.text }
        assertEquals(terminal.text, joined)
        return terminal
    }
}
