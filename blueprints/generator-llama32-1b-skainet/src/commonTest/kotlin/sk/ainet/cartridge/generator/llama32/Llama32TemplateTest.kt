package sk.ainet.cartridge.generator.llama32

import sk.ainet.cartridge.profiles.llm.LlmMessage
import sk.ainet.cartridge.profiles.llm.LlmRole
import kotlin.test.Test
import kotlin.test.assertEquals

class Llama32TemplateTest {

    /** Pure: the render must be byte-exact with the family template's no-tools path. */
    @Test
    fun rendersTheLlama3TurnStructure() {
        val rendered = renderLlama3(
            listOf(
                LlmMessage(LlmRole.SYSTEM, "You are a translator."),
                LlmMessage(LlmRole.USER, "Hello"),
            ),
        )
        assertEquals(
            "<|begin_of_text|>" +
                "<|start_header_id|>system<|end_header_id|>\n\nYou are a translator.<|eot_id|>" +
                "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>" +
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
            rendered,
        )
    }
}
