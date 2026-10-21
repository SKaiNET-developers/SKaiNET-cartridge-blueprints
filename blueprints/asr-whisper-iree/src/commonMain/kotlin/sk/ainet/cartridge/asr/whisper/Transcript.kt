package sk.ainet.cartridge.asr.whisper

/**
 * The tidy-up after a decode that stopped on a repetition. Greedy decoding over the short window can transcribe the
 * utterance and then start it again; the runtime's guard stops at the first repeated n-gram, but the repeat can begin
 * with a different token than the original (`F` vs ` F`), so a fragment of it may remain: `Fern sie ihr lauter. F`.
 */
public object Transcript {
    /**
     * [text] of a decode that stopped on a repetition, without the restart: the first sentence whose letters and
     * digits are a prefix of an earlier sentence's (the repeat, complete or cut short) is dropped with everything
     * after it, and so is an unterminated tail after a completed sentence — the guard cut the restart there, and a
     * restart that began with a variant token (`Gel` for `Geh`) is no literal prefix.
     */
    public fun dropRepeat(text: String): String {
        val sentences = SENTENCE.findAll(text).map { it.value }.toList()
        val kept = ArrayList<String>()
        val seen = ArrayList<String>()
        for (s in sentences) {
            val key = normalize(s)
            if (key.isEmpty()) continue
            if (seen.any { it.startsWith(key) }) break
            seen += key
            kept += s
        }
        if (kept.size > 1 && !kept.last().trimEnd().endsWithTerminator()) kept.removeAt(kept.lastIndex)
        return kept.joinToString("").trim()
    }

    private val SENTENCE = Regex("""[^.!?]*(?:[.!?]+|$)""")

    private fun normalize(s: String): String = s.lowercase().filter { it.isLetterOrDigit() }

    private fun String.endsWithTerminator(): Boolean = isNotEmpty() && last() in ".!?"
}
