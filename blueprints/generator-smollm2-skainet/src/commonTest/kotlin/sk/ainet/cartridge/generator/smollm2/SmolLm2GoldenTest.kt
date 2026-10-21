package sk.ainet.cartridge.generator.smollm2

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import sk.ainet.cartridge.profiles.llm.LlmEvent
import sk.ainet.cartridge.profiles.llm.LlmFinishReason
import sk.ainet.cartridge.profiles.llm.LlmMessage
import sk.ainet.cartridge.profiles.llm.LlmRole
import sk.ainet.cartridge.profiles.llm.LlmSessionConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.minutes

class SmolLm2GoldenTest {

    /** Pure: the render must be byte-exact with the checkpoint's chat_template. */
    @Test
    fun rendersTheCheckpointsChatTemplate() {
        val rendered = renderChatMl(GoldenCases.MESSAGES)
        assertEquals(
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, " +
                "trained by Hugging Face<|im_end|>\n" +
                "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n" +
                "<|im_start|>assistant\n",
            rendered,
        )
        // An explicit system message replaces the default, never doubles it.
        val custom = renderChatMl(
            listOf(
                LlmMessage(LlmRole.SYSTEM, "Answer in one word."),
                LlmMessage(LlmRole.USER, "Capital of France?"),
            ),
        )
        assertEquals(
            "<|im_start|>system\nAnswer in one word.<|im_end|>\n" +
                "<|im_start|>user\nCapital of France?<|im_end|>\n" +
                "<|im_start|>assistant\n",
            custom,
        )
    }

    @Test
    fun chatControlTokensEncodeAtomically() = runTest(timeout = 10.minutes) {
        withEngine("chatControlTokensEncodeAtomically") { engine ->
            assertEquals(
                listOf(SmolLm2GeneratorCartridge.IM_START_TOKEN_ID),
                engine.tokenizer.encode("<|im_start|>").toList(),
                "<|im_start|> must encode as the single added token, not as byte-level pieces",
            )
            assertEquals(
                listOf(SmolLm2GeneratorCartridge.IM_END_TOKEN_ID),
                engine.tokenizer.encode("<|im_end|>").toList(),
            )
            assertEquals(SmolLm2GeneratorCartridge.VOCAB_SIZE, engine.tokenizer.vocabSize)
        }
    }

    @Test
    fun greedyTranscriptIsTheGoldenOne() = runTest(timeout = 10.minutes) {
        withEngine("greedyTranscriptIsTheGoldenOne") { engine ->
            val session = engine.session(
                LlmSessionConfig(temperature = 0f, maxOutputTokens = GoldenCases.MAX_TOKENS),
            )
            session.prompt(GoldenCases.MESSAGES)
            val events = session.events.toList()

            assertEquals(LlmEvent.Ready, events.first(), "ordering: Ready first")
            val final = assertIs<LlmEvent.Final>(events.last(), "ordering: Final terminal")
            val deltas = events.drop(1).dropLast(1).map { assertIs<LlmEvent.Delta>(it).text }
            assertTrue(deltas.isNotEmpty(), "a real decode emits deltas")
            assertEquals(final.text, deltas.joinToString(""), "Final is the deltas' concatenation")

            assertEquals(GoldenCases.GOLDEN_TOKENS, final.tokens, "greedy token ids drifted")
            assertEquals(GoldenCases.GOLDEN_TEXT, final.text, "greedy transcript drifted")
            assertTrue(
                final.finishReason == LlmFinishReason.STOP || final.finishReason == LlmFinishReason.LENGTH,
                "greedy decode finishes by stop token or budget, got ${final.finishReason}",
            )
        }
    }

    @Test
    fun cancelYieldsPartialFinal() = runTest(timeout = 10.minutes) {
        withEngine("cancelYieldsPartialFinal") { engine ->
            val session = engine.session(
                LlmSessionConfig(temperature = 0f, maxOutputTokens = GoldenCases.MAX_TOKENS),
            )
            session.prompt(GoldenCases.MESSAGES)
            val events = mutableListOf<LlmEvent>()
            var deltasSeen = 0
            session.events.collect { event ->
                events += event
                if (event is LlmEvent.Delta && ++deltasSeen == 2) session.cancel()
            }
            val final = assertIs<LlmEvent.Final>(events.last())
            assertEquals(LlmFinishReason.CANCELLED, final.finishReason)
            assertTrue(final.text.isNotEmpty(), "cancel keeps the partial output")
            assertTrue(
                GoldenCases.GOLDEN_TEXT.startsWith(final.text),
                "the partial output is a prefix of the greedy transcript",
            )
        }
    }

    private suspend fun withEngine(testName: String, block: suspend (SmolLm2Engine) -> Unit) {
        val engine = try {
            SmolLm2GeneratorCartridge.open()
        } catch (failure: Exception) {
            println("SKIPPED $testName: ${failure.message}")
            return
        }
        try {
            block(engine)
        } finally {
            engine.close()
        }
    }
}
