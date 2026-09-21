package sk.ainet.cartridge.nlu.functiongemma

import android.util.Log
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.io.AndroidRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.transformers.gemma.iree.FunctionGemmaOfficialToolCallParserStrategy
import sk.ainet.transformers.iree.android.IreeKvSession
import sk.ainet.transformers.iree.android.IreeKvSpec
import java.io.File

/**
 * FunctionGemma-270M tool calling under the KV-cache contract: [warmUp] prefills the catalog prefix once and
 * snapshots the cache; every [resolve] restores the snapshot, runs the utterance as chunk calls and decodes
 * greedily until `<end_function_call>`, `<end_of_turn>`, the budget, or [Config.maxNewTokens].
 *
 * Everything model-specific comes from the materialized [pack]: graphs, parameter archives, tokenizer, and the
 * tool catalog. The runtime (`IreeKvSession`, `libskainet_iree_kv.so`), the chat template and the tool-call parser
 * are SKaiNET-transformers' released artifacts.
 *
 * @param cacheDir where the tokenized catalog prefix is cached between runs (an app's `filesDir` or `cacheDir`).
 */
public class FunctionGemmaNluCartridge(
    private val pack: PackDir,
    private val cacheDir: File,
    private val config: Config = Config(),
) : NluToolCallCartridge {

    public data class Config(
        /** IREE device; `null` = from the descriptor's target (`vulkan` when it names an accelerator, else `local-task`). */
        val device: String? = null,
        /** Tokens per prefill-with-past call; must equal the chunk the graphs were exported with. */
        val chunk: Int = 32,
        /** Sequence length of the prefill-at graph; the rendered catalog prefix must fit. */
        val prefillSeq: Int = 1024,
        val maxNewTokens: Int = 32,
        /** A prose answer (first token is not `<start_function_call>`) can never become a call: stop after this many tokens. */
        val maxProseTokens: Int = 8,
    )

    public val catalog: ToolCatalog = ToolCatalog.load(pack.toolCatalog)
    override val id: String = pack.id
    override val toolNames: Set<String> = catalog.names

    private val tag = "FunctionGemmaNlu"
    private val device = config.device ?: pack.device
    private val parser = FunctionGemmaOfficialToolCallParserStrategy()
    private var tokenizer: Tokenizer? = null
    private var session: IreeKvSession? = null
    private var prefix: IreeKvSession.Snapshot? = null
    private var prompt: FunctionGemmaPrompt? = null
    private var stopIds: Set<Int> = setOf(FunctionGemmaPrompt.EOS)
    private var startCallId: Int = -1

    /** Warm-up stage times in ms (tokenizer, prompt render + encode, session open, prefill). */
    public var warmUpTiming: Map<String, Long> = emptyMap()
        private set

    /** Prefix token count after [warmUp] — report it as the descriptor's `attributes.prompt_prefix_tokens`. */
    public var prefixTokens: Int = 0
        private set

    @Synchronized
    override fun warmUp() {
        if (prefix != null) return
        val timing = LinkedHashMap<String, Long>()
        val t0 = System.nanoTime()
        var t = t0
        fun lap(name: String) { timing[name] = (System.nanoTime() - t) / 1_000_000; t = System.nanoTime() }

        val gguf = pack.tokenizerGguf
        val fields = StreamingGGUFReader.open(AndroidRandomAccessSource.open(gguf.absolutePath)).use { it.fields }
        val tok = TokenizerFactory.fromGgufFields(fields)
        tokenizer = tok
        lap("tokenizer")

        val p = FunctionGemmaPrompt(catalog.definitions())
        prompt = p
        // Encoding a multi-kilobyte catalog prefix is slow on ART (tokenizer speed, not the model). The ids are a pure
        // function of (prefix text, tokenizer), so they are cached after the first warm-up.
        val cacheKey = (p.prefixText.hashCode().toLong() shl 32 xor gguf.length()).toString(16)
        val cacheFile = File(cacheDir, "nlu-functiongemma-prefix-$cacheKey.ids")
        val prefixIds = runCatching {
            if (cacheFile.exists()) cacheFile.readText().split(',').map { it.toInt() }.toIntArray() else null
        }.getOrNull() ?: (intArrayOf(FunctionGemmaPrompt.BOS) + tok.encode(p.prefixText)).also { ids ->
            runCatching { cacheFile.parentFile?.mkdirs(); cacheFile.writeText(ids.joinToString(",")) }
        }
        // The model keeps generating after a call (`<end_function_call><start_function_call>…`): stop at the
        // first closed call, not only at <end_of_turn>.
        stopIds = setOf(FunctionGemmaPrompt.EOS) + listOfNotNull(tok.encode("<end_function_call>").lastOrNull())
        startCallId = tok.encode("<start_function_call>").lastOrNull() ?: -1
        lap("prompt")

        require(prefixIds.size < config.prefillSeq) {
            "catalog '${catalog.id}' renders to ${prefixIds.size} prefix tokens, which does not fit the prefill graph " +
                "(${config.prefillSeq}). Shorten the catalog, or materialize with a longer prefill sequence."
        }
        val s = IreeKvSession(
            IreeKvSpec.functionGemma270m(config.chunk), device,
            pack.graph("with-past").path, pack.parameters("with-past").path,
            pack.graph("prefill-with-past").path, pack.parameters("prefill-with-past").path,
            pack.graph("prefill-at").path, pack.parameters("prefill-at").path,
        )
        session = s
        lap("open")

        val padded = IntArray(config.prefillSeq).also { prefixIds.copyInto(it) }
        s.prefill(padded, prefixIds.size)
        prefix = s.snapshot()
        s.releasePrefill()
        lap("prefill")

        prefixTokens = prefixIds.size
        warmUpTiming = timing
        Log.i(tag, "warm: catalog=${catalog.id}, prefix=${prefixIds.size} tokens, device=$device, chunk=${config.chunk}, ${(System.nanoTime() - t0) / 1_000_000} ms; stages=$timing")
    }

    @Synchronized
    override fun resolve(transcript: String, budgetMs: Long): NluResolution {
        val s = session; val snap = prefix; val tok = tokenizer; val p = prompt
        if (s == null || snap == null || tok == null || p == null) return NluResolution.Failed("not warmed up — call warmUp() first")
        val deadline = System.nanoTime() + budgetMs * 1_000_000
        var t = System.nanoTime()
        val ids = tok.encode(p.utteranceText(transcript.trim()))
        val tokenizeMs = (System.nanoTime() - t) / 1_000_000
        try {
            t = System.nanoTime()
            s.restore(snap)
            val restoreMs = (System.nanoTime() - t) / 1_000_000
            t = System.nanoTime()
            var next = -1
            var off = 0
            while (off < ids.size) {
                val n = minOf(config.chunk, ids.size - off)
                val buf = IntArray(config.chunk).also { ids.copyInto(it, 0, off, off + n) }
                next = s.chunk(buf, n)
                off += n
            }
            val chunkMs = (System.nanoTime() - t) / 1_000_000
            t = System.nanoTime()
            val out = ArrayList<Int>(config.maxNewTokens)
            var prose = false
            while (out.size < config.maxNewTokens) {
                if (next == FunctionGemmaPrompt.EOS) break
                out += next
                if (next in stopIds) break   // `<end_function_call>` closes the call: nothing after it is used
                if (out.size == 1 && next != startCallId) prose = true
                if (prose && out.size >= config.maxProseTokens) break   // prose never becomes a call; enough to report NoCall
                if (System.nanoTime() > deadline) {
                    val timing = NluResolution.Timing(tokenizeMs, restoreMs, chunkMs, (System.nanoTime() - t) / 1_000_000, out.size)
                    return NluResolution.Failed("budget of $budgetMs ms exceeded after ${out.size} tokens", timing = timing)
                }
                next = s.step(next)
            }
            val decodeMs = (System.nanoTime() - t) / 1_000_000
            val timing = NluResolution.Timing(tokenizeMs, restoreMs, chunkMs, decodeMs, out.size)
            val text = if (out.isEmpty()) "" else tok.decode(out.toIntArray())
            val calls = runCatching { parser.parse(text) }.getOrDefault(emptyList())
            val call = calls.firstOrNull() ?: return NluResolution.NoCall(text, timing)
            val name = catalog.snapName(call.name) ?: return NluResolution.NoCall(text, timing)
            val args = call.arguments.entries.associate { (k, v) -> k to v.toString().trim('"') }
            return NluResolution.Call(name, args, text, timing)
        } catch (e: IllegalStateException) {
            Log.e(tag, "resolve failed: ${e.message}")
            return NluResolution.Failed("runtime: ${e.message}", e)
        }
    }

    @Synchronized
    override fun close() {
        prefix?.close(); prefix = null
        session?.close(); session = null
        tokenizer = null
    }
}
