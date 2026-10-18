package sk.ainet.cartridge.nlu.qwen

import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.ChatTemplate
import sk.ainet.apps.kllama.chat.Qwen25ChatTemplate
import sk.ainet.apps.kllama.chat.QwenChatTemplate
import sk.ainet.apps.kllama.chat.ToolDefinition

/** The official chat template a flavor was trained on — SKaiNET-transformers' released implementations, not copies. */
public enum class QwenTemplate(public val id: String) {
    /** Qwen3 with thinking disabled: the generation prompt pre-fills an empty `<think>` block, as the official template does. */
    QWEN3_NO_THINKING("qwen3-no-thinking"),

    /** Qwen2.5-Instruct: the default persona, no thinking mode at all. */
    QWEN25("qwen25");

    public fun template(): ChatTemplate = when (this) {
        QWEN3_NO_THINKING -> QwenChatTemplate(enableThinking = false)
        QWEN25 -> Qwen25ChatTemplate()
    }

    public companion object {
        public fun byName(id: String): QwenTemplate = entries.firstOrNull { it.id == id }
            ?: throw IllegalArgumentException("unknown chat template '$id' (known: ${entries.joinToString { it.id }})")
    }
}

/**
 * Renders the Qwen prompt for a catalog and splits it into the part that is constant per catalog (the **prefix**,
 * prefilled once and snapshotted) and the part that changes per turn (the **utterance suffix**). The split is on the
 * `<|im_start|>user\n` boundary, so the two halves tokenise independently (special tokens are tokenizer boundaries)
 * and `prefix + utterance` equals the one-shot rendering token for token.
 *
 * The assistant turn is opened with [ASSISTANT_PREFIX], the start of a Hermes tool call up to the function name. The
 * model then only has to produce the name and the arguments: it cannot answer in prose first, and it skips the tokens
 * of the call's fixed opening. The engine re-attaches the prefix before parsing.
 */
public class QwenPrompt(catalog: List<ToolDefinition>, template: QwenTemplate) {
    public val prefixText: String
    private val suffixText: String

    init {
        val marker = "\u0000UTTERANCE\u0000"
        val full = template.template().apply(listOf(ChatMessage(ChatRole.USER, marker)), catalog, true)
        val at = full.indexOf(marker)
        require(at > 0) { "template rendering did not contain the utterance marker" }
        prefixText = full.substring(0, at)
        suffixText = full.substring(at + marker.length)
        require(prefixText.endsWith("<|im_start|>user\n")) { "unexpected prefix boundary: …${prefixText.takeLast(40)}" }
    }

    /** The per-turn text: the utterance closed as a user turn, the opened assistant turn, and the forced call opening. */
    public fun utteranceText(utterance: String): String = utterance + suffixText + ASSISTANT_PREFIX

    public companion object {
        /** The forced start of the assistant turn: a Hermes tool call up to the opening quote of the function name. */
        public const val ASSISTANT_PREFIX: String = "<tool_call>\n{\"name\": \""

        /** `<|im_end|>` — the end of the assistant turn. */
        public const val IM_END: Int = 151645

        /** `<|endoftext|>` — end of document. */
        public const val END_OF_TEXT: Int = 151643
    }
}
