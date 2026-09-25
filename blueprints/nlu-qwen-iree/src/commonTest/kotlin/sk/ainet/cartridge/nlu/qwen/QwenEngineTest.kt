package sk.ainet.cartridge.nlu.qwen

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** The whole resolve loop, on every target, against a scripted runtime: no model, no device. */
class QwenEngineTest {

    private val catalog = ToolCatalog.parse(
        """{"id":"toy-tools.v1","functions":[
             {"name":"toggle_lamp","description":"Switch the lamp.","parameters":{"state":{"type":"string","description":"on or off","enum":["on","off"]}},"required":["state"]},
             {"name":"roll_dice","description":"Roll a die."}]}""",
    )

    /** One id per word, JSON punctuation mark or special token, like the real tokenizer's single-token specials. */
    private class WordTokenizer : NluTokenizer {
        private val ids = LinkedHashMap<String, Int>().apply { put("<|im_end|>", QwenPrompt.IM_END); put("<|endoftext|>", QwenPrompt.END_OF_TEXT) }
        private val words = HashMap<Int, String>().apply { put(QwenPrompt.IM_END, "<|im_end|>"); put(QwenPrompt.END_OF_TEXT, "<|endoftext|>") }
        fun id(word: String): Int = ids.getOrPut(word) { (1000 + ids.size).also { words[it] = word } }
        override fun encode(text: String): IntArray =
            Regex("""<\|?/?[a-z_]+\|?>|[{}"\[\]:,\\]|[^<\s{}"\[\]:,\\]+|\s+""").findAll(text).map { id(it.value) }.toList().toIntArray()
        override fun decode(tokens: IntArray): String = tokens.joinToString("") { words.getValue(it) }
    }

    /** Emits a fixed token sequence, one per call, and records how it was driven. */
    private class ScriptedBackend(private val script: IntArray, private val failAtStep: Int = -1) : KvBackend {
        val calls = ArrayList<String>()
        private var cursor = 0
        var closedSnapshots = 0
        var steps = 0
        private fun next(): Int = script.getOrElse(cursor++) { QwenPrompt.IM_END }
        override fun prefill(tokens: IntArray, n: Int): Int { calls += "prefill(${tokens.size},$n)"; return 0 }
        override fun releasePrefill() { calls += "releasePrefill" }
        override fun snapshot(): KvBackend.Snapshot { calls += "snapshot"; return object : KvBackend.Snapshot { override fun close() { closedSnapshots++ } } }
        override fun restore(snapshot: KvBackend.Snapshot) { calls += "restore"; cursor = 0 }
        // Every chunk call predicts a next token; only the last one is used. The script starts after the final chunk.
        override fun chunk(tokens: IntArray, n: Int): Int { calls += "chunk(${tokens.size},$n)"; cursor = 0; return next() }
        override fun step(token: Int): Int { if (steps == failAtStep) throw IllegalStateException("device lost"); steps++; calls += "step"; return next() }
        override fun close() { calls += "close" }
    }

    private fun engine(tok: WordTokenizer, backend: ScriptedBackend, config: QwenEngine.Config = QwenEngine.Config(chunk = 4, prefillSeq = 512), cache: PrefixIdCache = PrefixIdCache.None) =
        QwenEngine("toy-cartridge", catalog, QwenTemplate.QWEN3_NO_THINKING, config, { tok }, { backend }, cache)

    /** The model's continuation after the forced `<tool_call>\n{"name": "`. */
    private fun WordTokenizer.script(continuation: String): IntArray = encode(continuation)

    @Test
    fun `a closed call resolves to the catalog function and string arguments and decoding stops at its brace`() {
        val tok = WordTokenizer()
        // The model closes the call and then starts a duplicate one: nothing after the first closing brace is decoded.
        val backend = ScriptedBackend(tok.script("""toggle_lamp", "arguments": {"state": "off"}}
</tool_call>
<tool_call>
{"name": "roll_dice", "arguments": {}}"""))
        val e = engine(tok, backend)
        e.warmUp()
        val r = assertIs<NluResolution.Call>(e.resolve("  turn the lamp off  ", budgetMs = 10_000))
        assertEquals("toggle_lamp", r.name)
        assertEquals(mapOf("state" to "off"), r.args)
        assertTrue(r.raw.endsWith("{\"state\": \"off\"}}\n</tool_call>"), r.raw)
        assertEquals(tok.script("""toggle_lamp", "arguments": {"state": "off"}}""").size, r.timing.decodeTokens, "stopped at the closing brace")
        assertEquals(setOf("toggle_lamp", "roll_dice"), e.toolNames)
    }

    @Test
    fun `braces inside argument strings do not end the call`() {
        val tok = WordTokenizer()
        val backend = ScriptedBackend(tok.script("""toggle_lamp", "arguments": {"state": "o}n"}}"""))
        val e = engine(tok, backend); e.warmUp()
        assertEquals(mapOf("state" to "o}n"), assertIs<NluResolution.Call>(e.resolve("x", 10_000)).args)
    }

    @Test
    fun `the utterance carries the forced call opening and no prefix is re-fed`() {
        val tok = WordTokenizer()
        val backend = ScriptedBackend(tok.script("""roll_dice", "arguments": {}}"""))
        val e = engine(tok, backend)
        e.warmUp(); e.warmUp()
        assertEquals(listOf("prefill(512,${e.prefixTokens})", "snapshot", "releasePrefill"), backend.calls.toList())
        assertTrue(e.prefixTokens > 10)

        assertIs<NluResolution.Call>(e.resolve("roll the dice please now", 10_000))
        val afterWarmUp = backend.calls.drop(3)
        assertEquals("restore", afterWarmUp.first())
        val chunks = afterWarmUp.filter { it.startsWith("chunk") }
        assertTrue(chunks.size >= 2 && chunks.all { it.startsWith("chunk(4,") }, "utterance is fed in fixed-size chunks: $chunks")
        val fed = tok.encode(QwenPrompt(catalog.definitions(), QwenTemplate.QWEN3_NO_THINKING).utteranceText("roll the dice please now"))
        assertEquals((fed.size + 3) / 4, chunks.size, "the whole per-turn text, forced opening included, is chunked")

        e.close()
        assertEquals("close", backend.calls.last())
        assertEquals(1, backend.closedSnapshots)
        assertIs<NluResolution.Failed>(e.resolve("again", 10_000))
    }

    @Test
    fun `a near-miss name is snapped and an unknown one is no call`() {
        val tok = WordTokenizer()
        assertEquals("toggle_lamp", assertIs<NluResolution.Call>(engine(tok, ScriptedBackend(tok.script("""Toggle__Lamp", "arguments": {"state": "on"}}"""))).apply { warmUp() }.resolve("x", 10_000)).name)
        assertIs<NluResolution.NoCall>(engine(tok, ScriptedBackend(tok.script("""open_garage", "arguments": {}}"""))).apply { warmUp() }.resolve("x", 10_000))
    }

    @Test
    fun `a call that never closes is no call and end of turn stops decoding`() {
        val tok = WordTokenizer()
        val runaway = engine(tok, ScriptedBackend(tok.script("""toggle_lamp", "arguments": {"state": "on and on and on and on and on""")),
            QwenEngine.Config(chunk = 4, prefillSeq = 512, maxNewTokens = 6))
        runaway.warmUp()
        assertEquals(6, assertIs<NluResolution.NoCall>(runaway.resolve("x", 10_000)).timing.decodeTokens)

        val ended = engine(tok, ScriptedBackend(tok.script("toggle_lamp") + QwenPrompt.IM_END))
        ended.warmUp()
        val r = assertIs<NluResolution.NoCall>(ended.resolve("x", 10_000))
        assertEquals(1, r.timing.decodeTokens)
        assertTrue(r.text.startsWith(QwenPrompt.ASSISTANT_PREFIX))
    }

    @Test
    fun `an exhausted budget and a runtime failure are reported and never thrown`() {
        val tok = WordTokenizer()
        val slow = engine(tok, ScriptedBackend(tok.script("""roll_dice", "arguments": {}}""")))
        slow.warmUp()
        assertTrue(assertIs<NluResolution.Failed>(slow.resolve("roll", budgetMs = -1)).reason.contains("budget"))

        val broken = engine(tok, ScriptedBackend(tok.script("""roll_dice", "arguments": {}}"""), failAtStep = 1))
        broken.warmUp()
        assertTrue(assertIs<NluResolution.Failed>(broken.resolve("roll", 10_000)).reason.contains("device lost"))
    }

    @Test
    fun `a catalog that does not fit the prefill graph is refused with the numbers`() {
        val tok = WordTokenizer()
        val e = engine(tok, ScriptedBackend(IntArray(0)), QwenEngine.Config(chunk = 4, prefillSeq = 8))
        val error = assertFailsWith<IllegalArgumentException> { e.warmUp() }
        assertTrue(error.message!!.contains("does not fit the prefill graph (8)"), error.message)
    }

    @Test
    fun `the tokenized prefix is cached by prefix text`() {
        val tok = WordTokenizer()
        val store = HashMap<String, IntArray>()
        val cache = object : PrefixIdCache {
            override fun get(key: String) = store[key]
            override fun put(key: String, ids: IntArray) { store[key] = ids }
        }
        engine(tok, ScriptedBackend(IntArray(0)), cache = cache).warmUp()
        assertEquals(1, store.size)
        val counting = object : NluTokenizer {
            var prefixEncodes = 0
            override fun encode(text: String): IntArray { if (text.length > 200) prefixEncodes++; return tok.encode(text) }
            override fun decode(tokens: IntArray) = tok.decode(tokens)
        }
        QwenEngine("toy", catalog, QwenTemplate.QWEN3_NO_THINKING, QwenEngine.Config(4, 512), { counting }, { ScriptedBackend(IntArray(0)) }, cache).warmUp()
        assertEquals(0, counting.prefixEncodes, "a cached prefix must not be re-encoded")
    }
}
