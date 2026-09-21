package sk.ainet.cartridge.nlu.functiongemma

import sk.ainet.transformers.gemma.iree.FunctionGemmaOfficialToolCallParserStrategy
import kotlin.time.TimeSource

/**
 * FunctionGemma-270M tool calling under the KV-cache contract, platform-independent: [warmUp] prefills the catalog
 * prefix once and snapshots the cache; every [resolve] restores the snapshot, runs the utterance as chunk calls and
 * decodes greedily until `<end_function_call>`, `<end_of_turn>`, the budget, or [Config.maxNewTokens].
 *
 * What differs per platform is injected: the [KvBackend] (the runtime), the [NluTokenizer] and the [PrefixIdCache].
 * The chat template and the tool-call parser are SKaiNET-transformers' released common code.
 *
 * Not thread-safe. Platform bindings that promise otherwise wrap it.
 */
public class FunctionGemmaEngine(
    override val id: String,
    public val catalog: ToolCatalog,
    private val config: Config = Config(),
    private val openTokenizer: () -> NluTokenizer,
    private val openBackend: () -> KvBackend,
    private val prefixCache: PrefixIdCache = PrefixIdCache.None,
    /** Distinguishes tokenizers in the prefix cache key (e.g. the tokenizer file's size); the prefix text is always part of it. */
    private val cacheSalt: String = "",
    private val log: (String) -> Unit = {},
) : NluToolCallCartridge {

    public data class Config(
        /** Tokens per prefill-with-past call; must equal the chunk the graphs were exported with. */
        val chunk: Int = 32,
        /** Sequence length of the prefill-at graph; the rendered catalog prefix must fit. */
        val prefillSeq: Int = 1024,
        val maxNewTokens: Int = 32,
        /** A prose answer (first token is not `<start_function_call>`) can never become a call: stop after this many tokens. */
        val maxProseTokens: Int = 8,
    )

    override val toolNames: Set<String> = catalog.names

    private val parser = FunctionGemmaOfficialToolCallParserStrategy()
    private var tokenizer: NluTokenizer? = null
    private var backend: KvBackend? = null
    private var prefix: KvBackend.Snapshot? = null
    private var prompt: FunctionGemmaPrompt? = null
    private var stopIds: Set<Int> = setOf(FunctionGemmaPrompt.EOS)
    private var startCallId: Int = -1

    /** Warm-up stage times in ms (tokenizer, prompt render + encode, backend open, prefill). */
    public var warmUpTiming: Map<String, Long> = emptyMap()
        private set

    /** Prefix token count after [warmUp] — report it as the descriptor's `attributes.prompt_prefix_tokens`. */
    public var prefixTokens: Int = 0
        private set

    override fun warmUp() {
        if (prefix != null) return
        val timing = LinkedHashMap<String, Long>()
        val start = TimeSource.Monotonic.markNow()
        var lapStart = start
        fun lap(name: String) { timing[name] = lapStart.elapsedNow().inWholeMilliseconds; lapStart = TimeSource.Monotonic.markNow() }

        val tok = openTokenizer()
        tokenizer = tok
        lap("tokenizer")

        val p = FunctionGemmaPrompt(catalog.definitions())
        prompt = p
        // The ids are a pure function of (prefix text, tokenizer): cache them, keyed by both.
        val key = "${p.prefixText.hashCode().toUInt().toString(16)}-${p.prefixText.length.toString(16)}-$cacheSalt"
        val prefixIds = prefixCache.get(key) ?: (intArrayOf(FunctionGemmaPrompt.BOS) + tok.encode(p.prefixText)).also { prefixCache.put(key, it) }
        // The model keeps generating after a call (`<end_function_call><start_function_call>…`): stop at the
        // first closed call, not only at <end_of_turn>.
        stopIds = setOf(FunctionGemmaPrompt.EOS) + listOfNotNull(tok.encode("<end_function_call>").lastOrNull())
        startCallId = tok.encode("<start_function_call>").lastOrNull() ?: -1
        lap("prompt")

        require(prefixIds.size < config.prefillSeq) {
            "catalog '${catalog.id}' renders to ${prefixIds.size} prefix tokens, which does not fit the prefill graph " +
                "(${config.prefillSeq}). Shorten the catalog, or materialize with a longer prefill sequence."
        }
        val b = openBackend()
        backend = b
        lap("open")

        b.prefill(IntArray(config.prefillSeq).also { prefixIds.copyInto(it) }, prefixIds.size)
        prefix = b.snapshot()
        b.releasePrefill()
        lap("prefill")

        prefixTokens = prefixIds.size
        warmUpTiming = timing
        log("warm: catalog=${catalog.id}, prefix=${prefixIds.size} tokens, chunk=${config.chunk}, ${start.elapsedNow().inWholeMilliseconds} ms; stages=$timing")
    }

    override fun resolve(transcript: String, budgetMs: Long): NluResolution {
        val b = backend; val snap = prefix; val tok = tokenizer; val p = prompt
        if (b == null || snap == null || tok == null || p == null) return NluResolution.Failed("not warmed up — call warmUp() first")
        val started = TimeSource.Monotonic.markNow()
        var mark = started
        fun lapMs(): Long = mark.elapsedNow().inWholeMilliseconds.also { mark = TimeSource.Monotonic.markNow() }

        val ids = tok.encode(p.utteranceText(transcript.trim()))
        val tokenizeMs = lapMs()
        try {
            b.restore(snap)
            val restoreMs = lapMs()

            var next = -1
            var off = 0
            while (off < ids.size) {
                val n = minOf(config.chunk, ids.size - off)
                next = b.chunk(IntArray(config.chunk).also { ids.copyInto(it, 0, off, off + n) }, n)
                off += n
            }
            val chunkMs = lapMs()

            val out = ArrayList<Int>(config.maxNewTokens)
            var prose = false
            while (out.size < config.maxNewTokens) {
                if (next == FunctionGemmaPrompt.EOS) break
                out += next
                if (next in stopIds) break   // `<end_function_call>` closes the call: nothing after it is used
                if (out.size == 1 && next != startCallId) prose = true
                if (prose && out.size >= config.maxProseTokens) break   // prose never becomes a call; enough to report NoCall
                if (started.elapsedNow().inWholeMilliseconds > budgetMs) {
                    val timing = NluResolution.Timing(tokenizeMs, restoreMs, chunkMs, mark.elapsedNow().inWholeMilliseconds, out.size)
                    return NluResolution.Failed("budget of $budgetMs ms exceeded after ${out.size} tokens", timing = timing)
                }
                next = b.step(next)
            }
            val timing = NluResolution.Timing(tokenizeMs, restoreMs, chunkMs, lapMs(), out.size)

            val text = if (out.isEmpty()) "" else tok.decode(out.toIntArray())
            val call = runCatching { parser.parse(text) }.getOrDefault(emptyList()).firstOrNull()
                ?: return NluResolution.NoCall(text, timing)
            val name = catalog.snapName(call.name) ?: return NluResolution.NoCall(text, timing)
            val args = call.arguments.entries.associate { (k, v) -> k to v.toString().trim('"') }
            return NluResolution.Call(name, args, text, timing)
        } catch (e: IllegalStateException) {
            log("resolve failed: ${e.message}")
            return NluResolution.Failed("runtime: ${e.message}", e)
        }
    }

    override fun close() {
        prefix?.close(); prefix = null
        backend?.close(); backend = null
        tokenizer = null
    }
}
