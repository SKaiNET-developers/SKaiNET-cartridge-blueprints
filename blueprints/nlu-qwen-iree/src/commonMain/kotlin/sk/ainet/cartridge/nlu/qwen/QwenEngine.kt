package sk.ainet.cartridge.nlu.qwen

import kotlinx.serialization.json.JsonPrimitive
import sk.ainet.apps.kllama.chat.ToolCallParser
import kotlin.time.TimeSource

/**
 * Qwen tool calling under the `qwen-kv-v1` KV-cache contract, platform-independent: [warmUp] prefills the catalog
 * prefix once and snapshots the cache; every [resolve] restores the snapshot, runs the utterance (with the forced
 * call opening, [QwenPrompt.ASSISTANT_PREFIX]) as chunk calls and decodes greedily until the call's closing brace,
 * an end-of-turn token, the budget, or [Config.maxNewTokens].
 *
 * What differs per platform is injected: the [KvBackend] (the runtime), the [NluTokenizer] and the [PrefixIdCache].
 * The chat templates and the tool-call JSON parser are SKaiNET-transformers' released common code.
 *
 * Not thread-safe. Platform bindings that promise otherwise wrap it.
 */
public class QwenEngine(
    override val id: String,
    public val catalog: ToolCatalog,
    private val template: QwenTemplate,
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
        /** A call that has not closed after this many tokens is reported as no call. */
        val maxNewTokens: Int = 48,
    )

    override val toolNames: Set<String> = catalog.names

    private var tokenizer: NluTokenizer? = null
    private var backend: KvBackend? = null
    private var prefix: KvBackend.Snapshot? = null
    private var prompt: QwenPrompt? = null

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

        val p = QwenPrompt(catalog.definitions(), template)
        prompt = p
        // The ids are a pure function of (prefix text, tokenizer): cache them, keyed by both. Qwen adds no BOS.
        val key = "${p.prefixText.hashCode().toUInt().toString(16)}-${p.prefixText.length.toString(16)}-$cacheSalt"
        val prefixIds = prefixCache.get(key) ?: tok.encode(p.prefixText).also { prefixCache.put(key, it) }
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
        log("warm: catalog=${catalog.id}, template=${template.id}, prefix=${prefixIds.size} tokens, chunk=${config.chunk}, ${start.elapsedNow().inWholeMilliseconds} ms; stages=$timing")
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
            var callLength = -1
            while (out.size < config.maxNewTokens) {
                if (next == QwenPrompt.IM_END || next == QwenPrompt.END_OF_TEXT) break
                out += next
                callLength = CallScanner.completeLength(tok.decode(out.toIntArray()))
                if (callLength >= 0) break   // the call is closed: whatever the model would add after it is never used
                if (started.elapsedNow().inWholeMilliseconds > budgetMs) {
                    val timing = NluResolution.Timing(tokenizeMs, restoreMs, chunkMs, mark.elapsedNow().inWholeMilliseconds, out.size)
                    return NluResolution.Failed("budget of $budgetMs ms exceeded after ${out.size} tokens", timing = timing)
                }
                next = b.step(next)
            }
            val timing = NluResolution.Timing(tokenizeMs, restoreMs, chunkMs, lapMs(), out.size)

            val continuation = if (out.isEmpty()) "" else tok.decode(out.toIntArray())
            val raw = QwenPrompt.ASSISTANT_PREFIX + continuation
            if (callLength < 0) return NluResolution.NoCall(raw, timing)
            val json = QwenPrompt.ASSISTANT_PREFIX.substringAfter("<tool_call>").trimStart() + continuation.substring(0, callLength)
            val call = ToolCallParser.parseJsonToolCall(json) ?: return NluResolution.NoCall(raw, timing)
            val name = catalog.snapName(call.name) ?: return NluResolution.NoCall(raw, timing)
            val args = call.arguments.entries.associate { (k, v) -> k to ((v as? JsonPrimitive)?.content ?: v.toString()) }
            return NluResolution.Call(name, args, "<tool_call>\n$json\n</tool_call>", timing)
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
