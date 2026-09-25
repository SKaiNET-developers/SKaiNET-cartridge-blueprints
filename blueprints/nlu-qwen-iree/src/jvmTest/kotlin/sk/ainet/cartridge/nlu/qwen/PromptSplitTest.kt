package sk.ainet.cartridge.nlu.qwen

import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PromptSplitTest {
    private val catalog = ToolCatalog.load(kotlinx.io.files.Path(System.getProperty("blueprint.dir"), "samples/toy-catalog.json"))

    @Test
    fun `prefix plus utterance equals the one-shot rendering plus the forced call opening, for every template`() {
        for (t in QwenTemplate.entries) {
            val prompt = QwenPrompt(catalog.definitions(), t)
            val utterance = "turn the lamp off"
            val oneShot = t.template().apply(listOf(ChatMessage(ChatRole.USER, utterance)), catalog.definitions(), true)
            assertEquals(oneShot + QwenPrompt.ASSISTANT_PREFIX, prompt.prefixText + prompt.utteranceText(utterance), t.id)
        }
    }

    @Test
    fun `the prefix is constant per catalog, ends on the user-turn boundary and declares every function`() {
        for (t in QwenTemplate.entries) {
            val a = QwenPrompt(catalog.definitions(), t)
            assertEquals(a.prefixText, QwenPrompt(catalog.definitions(), t).prefixText)
            assertTrue(a.prefixText.endsWith("<|im_start|>user\n"), t.id)
            for (name in catalog.names) assertTrue(a.prefixText.contains(name), "${t.id}: the prefix must declare $name")
            assertTrue(a.utteranceText("x").endsWith("<|im_start|>assistant\n" + (if (t == QwenTemplate.QWEN3_NO_THINKING) "<think>\n\n</think>\n\n" else "") + QwenPrompt.ASSISTANT_PREFIX), t.id)
        }
    }

    @Test
    fun `thinking is disabled for Qwen3 and absent for Qwen2_5`() {
        assertTrue(QwenPrompt(catalog.definitions(), QwenTemplate.QWEN3_NO_THINKING).utteranceText("x").contains("<think>\n\n</think>"))
        val q25 = QwenPrompt(catalog.definitions(), QwenTemplate.QWEN25)
        assertFalse((q25.prefixText + q25.utteranceText("x")).contains("<think>"))
        assertTrue(q25.prefixText.contains("You are Qwen, created by Alibaba Cloud."), "Qwen2.5's official default persona")
    }

    @Test
    fun `template ids resolve and unknown ones are refused`() {
        assertEquals(QwenTemplate.QWEN3_NO_THINKING, QwenTemplate.byName("qwen3-no-thinking"))
        assertEquals(QwenTemplate.QWEN25, QwenTemplate.byName("qwen25"))
        kotlin.test.assertFailsWith<IllegalArgumentException> { QwenTemplate.byName("qwen4") }
    }
}
