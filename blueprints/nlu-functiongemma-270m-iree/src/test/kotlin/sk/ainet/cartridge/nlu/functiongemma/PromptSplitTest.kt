package sk.ainet.cartridge.nlu.functiongemma

import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.transformers.gemma.iree.FunctionGemmaOfficialChatTemplate
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PromptSplitTest {
    private val catalog = ToolCatalog.load(File(System.getProperty("blueprint.dir"), "samples/toy-catalog.json"))

    @Test
    fun `prefix plus utterance equals the one-shot rendering`() {
        val prompt = FunctionGemmaPrompt(catalog.definitions())
        val utterance = "turn the lamp off"
        val oneShot = FunctionGemmaOfficialChatTemplate().apply(listOf(ChatMessage(ChatRole.USER, utterance)), catalog.definitions(), true)
        assertEquals(oneShot, prompt.prefixText + prompt.utteranceText(utterance))
    }

    @Test
    fun `the prefix is constant per catalog and ends on the user-turn boundary`() {
        val a = FunctionGemmaPrompt(catalog.definitions())
        val b = FunctionGemmaPrompt(catalog.definitions())
        assertEquals(a.prefixText, b.prefixText)
        assertTrue(a.prefixText.endsWith("<start_of_turn>user\n"))
        for (name in catalog.names) assertTrue(a.prefixText.contains(name), "the prefix must declare $name")
        assertTrue(a.utteranceText("x").startsWith("x"))
    }
}
