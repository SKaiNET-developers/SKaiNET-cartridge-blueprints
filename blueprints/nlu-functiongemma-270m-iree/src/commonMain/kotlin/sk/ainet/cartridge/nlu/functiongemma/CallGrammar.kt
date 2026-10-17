package sk.ainet.cartridge.nlu.functiongemma

/**
 * The output tokens a catalog leaves no choice about. A FunctionGemma call reads
 * `<start_function_call>call:NAME{KEY:<escape>VALUE<escape>,…}<end_function_call>`: once the model has started a
 * call, `call:` follows; once a name's first token rules out every other function, the rest of the name and `{`
 * follow; after a key, `:` (and `<escape>` for enum parameters); after the first token of an enum value, the rest of
 * it and `<escape>`. The engine feeds such runs to the model in one chunk call instead of one step per token, and
 * stops at the `}` that closes the call.
 *
 * Only the model's own choices are kept as choices: which function, which keys in which order, every value, and
 * whether to close. Free values (no enum) are the model's entirely, including whether it quotes them with
 * `<escape>`. When the model emits a token the grammar did not expect, the cursor turns [Cursor.off] and forces
 * nothing more for that utterance — the call is then decoded exactly as without the grammar.
 *
 * Token sequences come from the cartridge's tokenizer, per catalog element (`call:set_light{`, `room:<escape>`,
 * `kitchen<escape>`); special tokens are tokenizer boundaries, so the splits match the model's own output.
 */
internal class CallGrammar(catalog: ToolCatalog, private val tok: NluTokenizer) {
    private fun single(text: String): Int = tok.encode(text).singleOrNull() ?: -1

    private val startCall = single("<start_function_call>")
    private val escape = single("<escape>")
    private val comma = single(",")
    private val close = single("}")

    private class Param(val key: IntArray, val values: List<IntArray>?)
    private class Function(val name: IntArray, val params: List<Param>)

    private val functions = catalog.functions.map { f ->
        Function(
            tok.encode("call:${f.name}{"),
            f.parameters.map { (k, p) ->
                val enum = p.enum?.takeIf { it.isNotEmpty() }
                Param(tok.encode(if (enum != null) "$k:<escape>" else "$k:"), enum?.map { tok.encode("$it<escape>") })
            },
        )
    }

    /** Every marker the grammar needs is a single token in this tokenizer; otherwise it forces nothing. */
    val usable: Boolean = listOf(startCall, escape, comma, close).all { it >= 0 }

    fun cursor(): Cursor = Cursor()

    private enum class Phase { START, NAME, KEY, ENUM, FREE, SEPARATOR, CLOSED, OFF }

    /** Follows one decoded call token by token. */
    inner class Cursor {
        /** Candidate token sequences of the element being decoded, and how many of their tokens are matched. */
        private var candidates: List<IntArray> = emptyList()
        private var matched = 0
        private var phase = if (usable) Phase.START else Phase.OFF
        private var function: Function? = null
        private val used = HashSet<Param>()
        private var param: Param? = null
        private var quoted = false

        val off: Boolean get() = phase == Phase.OFF
        /** The `}` of the call is decoded: nothing after it is used. */
        val closed: Boolean get() = phase == Phase.CLOSED

        /** The next token if the grammar leaves no choice, else -1. */
        fun forced(): Int = when (phase) {
            Phase.NAME, Phase.KEY, Phase.ENUM -> candidates.map { it[matched] }.distinct().singleOrNull() ?: -1
            Phase.SEPARATOR -> if (unused().isEmpty()) close else -1
            else -> -1
        }

        /** Advances over [token], the model's or a forced one. */
        fun accept(token: Int) {
            when (phase) {
                Phase.START -> if (token == startCall) element(Phase.NAME, functions.map { it.name }) else phase = Phase.OFF
                Phase.NAME, Phase.KEY, Phase.ENUM -> match(token)
                Phase.FREE -> free(token)
                Phase.SEPARATOR -> when (token) {
                    comma -> keys()
                    close -> phase = Phase.CLOSED
                    else -> phase = Phase.OFF
                }
                Phase.CLOSED, Phase.OFF -> {}
            }
        }

        private fun unused(): List<Param> = function?.params?.filter { it !in used }.orEmpty()

        private fun element(next: Phase, sequences: List<IntArray>) {
            phase = next; candidates = sequences; matched = 0
        }

        /** After `{` or `,`: a key of a parameter not given yet; after `{` also `}`. */
        private fun keys(allowClose: Boolean = false) {
            val keys = unused().map { it.key }
            element(Phase.KEY, if (allowClose) keys + intArrayOf(close) else keys)
            if (candidates.isEmpty()) phase = Phase.OFF
        }

        private fun match(token: Int) {
            candidates = candidates.filter { it.size > matched && it[matched] == token }
            matched++
            if (candidates.isEmpty()) { phase = Phase.OFF; return }
            val done = candidates.singleOrNull { it.size == matched } ?: return
            when (phase) {
                Phase.NAME -> { function = functions.first { it.name === done }; keys(allowClose = true) }
                Phase.KEY -> {
                    if (done.size == 1 && done[0] == close) { phase = Phase.CLOSED; return }
                    val p = unused().first { it.key === done }
                    used += p; param = p
                    val values = p.values
                    if (values != null) element(Phase.ENUM, values) else { phase = Phase.FREE; quoted = false; matched = 0 }
                }
                Phase.ENUM -> phase = Phase.SEPARATOR
                else -> {}
            }
        }

        /** A free value: `<escape>…<escape>`, or unquoted up to `,` or `}`. */
        private fun free(token: Int) {
            val first = matched++ == 0
            when {
                first && token == escape -> quoted = true
                quoted -> if (token == escape) phase = Phase.SEPARATOR
                token == comma -> keys()
                token == close -> phase = Phase.CLOSED
            }
        }
    }
}
