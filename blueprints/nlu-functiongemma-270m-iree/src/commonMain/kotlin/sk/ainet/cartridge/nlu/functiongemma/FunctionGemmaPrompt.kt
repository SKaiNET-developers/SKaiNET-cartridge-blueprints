package sk.ainet.cartridge.nlu.functiongemma

import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.ToolDefinition
import sk.ainet.transformers.gemma.iree.FunctionGemmaOfficialChatTemplate

/**
 * Renders the FunctionGemma prompt for a catalog and splits it into the part that is constant
 * per catalog (the **prefix**, prefilled once and snapshotted) and the part that changes per
 * turn (the **utterance suffix**). The split is on the `<start_of_turn>user\n` boundary of the
 * official template, so the two halves tokenise independently (special tokens are tokenizer
 * boundaries) and `prefix + utterance` equals the one-shot rendering token for token.
 *
 * The template itself is SKaiNET-transformers' `FunctionGemmaOfficialChatTemplate` — not a copy.
 */
public class FunctionGemmaPrompt(catalog: List<ToolDefinition>) {
    public val prefixText: String
    private val suffixText: String

    init {
        val marker = "\u0000UTTERANCE\u0000"
        val full = FunctionGemmaOfficialChatTemplate().apply(listOf(ChatMessage(ChatRole.USER, marker)), catalog, true)
        val at = full.indexOf(marker)
        require(at > 0) { "template rendering did not contain the utterance marker" }
        prefixText = full.substring(0, at)
        suffixText = full.substring(at + marker.length)
        require(prefixText.endsWith("<start_of_turn>user\n")) { "unexpected prefix boundary: …${prefixText.takeLast(40)}" }
    }

    /** The per-turn text: the utterance closed as a user turn and the opened model turn. */
    public fun utteranceText(utterance: String): String = utterance + suffixText

    public companion object {
        /** `<bos>` id of the Gemma tokenizer; the template does not emit it, the prefix must start with it. */
        public const val BOS: Int = 2

        /** `<end_of_turn>` — the model's end-of-answer token. */
        public const val EOS: Int = 106
    }
}
