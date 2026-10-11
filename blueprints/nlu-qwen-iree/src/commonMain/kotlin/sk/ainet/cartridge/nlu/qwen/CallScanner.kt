package sk.ainet.cartridge.nlu.qwen

/**
 * Finds where a forced Hermes call ends, on the decoded *text*. The call opens inside the function-name string of
 * `{"name": "` ([QwenPrompt.ASSISTANT_PREFIX]), so scanning starts one object deep and inside a string; the call is
 * complete at the brace that brings the depth back to zero. Braces inside JSON strings (and escaped quotes) are
 * ignored.
 *
 * Why text and not token ids: in this template the model does not reliably emit `</tool_call>`, `<|im_end|>` or
 * `</think>` as their dedicated vocabulary ids after a call — it often continues with a second, duplicate call — so
 * an id-based stop set lets it run on. The closing brace is always there.
 */
internal object CallScanner {
    /** Length of the complete call at the start of [text] (the continuation after the forced prefix), or -1. */
    fun completeLength(text: String): Int {
        var depth = 1
        var inString = true
        var escaped = false
        for ((i, c) in text.withIndex()) {
            if (inString) {
                when {
                    escaped -> escaped = false
                    c == '\\' -> escaped = true
                    c == '"' -> inString = false
                }
                continue
            }
            when (c) {
                '"' -> inString = true
                '{' -> depth++
                '}' -> { depth--; if (depth == 0) return i + 1 }
            }
        }
        return -1
    }
}
