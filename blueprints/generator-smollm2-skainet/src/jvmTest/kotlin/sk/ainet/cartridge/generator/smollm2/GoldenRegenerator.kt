package sk.ainet.cartridge.generator.smollm2

import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import sk.ainet.cartridge.profiles.llm.LlmEvent
import sk.ainet.cartridge.profiles.llm.LlmSessionConfig
import java.io.File
import kotlin.test.Test
import kotlin.time.Duration.Companion.minutes

/**
 * Deliberate golden regeneration, guarded by REGENERATE_GOLDENS (gradle
 * -PregenerateGoldens=true). Writes samples/golden-transcript.json and prints the paste-ready
 * Kotlin literals for [GoldenCases].
 */
class GoldenRegenerator {

    @Test
    fun regenerate() = runTest(timeout = 10.minutes) {
        if (System.getenv("REGENERATE_GOLDENS").isNullOrBlank()) {
            println("GoldenRegenerator: REGENERATE_GOLDENS not set, skipping")
            return@runTest
        }
        val engine = SmolLm2GeneratorCartridge.open()
        try {
            val session = engine.session(
                LlmSessionConfig(temperature = 0f, maxOutputTokens = GoldenCases.MAX_TOKENS),
            )
            session.prompt(GoldenCases.MESSAGES)
            val final = session.events.filterIsInstance<LlmEvent.Final>().first()

            println("GOLDEN_TEXT = ${kotlinLiteral(final.text)}")
            println("GOLDEN_TOKENS = listOf(${final.tokens!!.joinToString(", ")})")
            println("finishReason = ${final.finishReason}")

            val out = File("samples/golden-transcript.json")
            out.parentFile?.mkdirs()
            out.writeText(
                buildString {
                    appendLine("{")
                    appendLine("  \"model\": \"${SmolLm2GeneratorCartridge.MODEL_REPO}@${SmolLm2GeneratorCartridge.MODEL_REVISION}\",")
                    appendLine("  \"decode\": \"greedy, max_tokens ${GoldenCases.MAX_TOKENS}, eos ${SmolLm2GeneratorCartridge.IM_END_TOKEN_ID}\",")
                    appendLine("  \"messages\": [{\"role\": \"user\", \"content\": ${jsonString(GoldenCases.MESSAGES[0].content)}}],")
                    appendLine("  \"finish_reason\": \"${final.finishReason}\",")
                    appendLine("  \"tokens\": [${final.tokens!!.joinToString(", ")}],")
                    appendLine("  \"text\": ${jsonString(final.text)}")
                    appendLine("}")
                },
            )
            println("wrote ${out.absolutePath}")
        } finally {
            engine.close()
        }
    }

    private fun kotlinLiteral(s: String): String =
        "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\""

    private fun jsonString(s: String): String =
        "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\""
}
