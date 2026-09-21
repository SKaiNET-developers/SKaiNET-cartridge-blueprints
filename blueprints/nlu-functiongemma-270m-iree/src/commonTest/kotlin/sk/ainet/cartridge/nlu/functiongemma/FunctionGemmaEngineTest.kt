package sk.ainet.cartridge.nlu.functiongemma

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** The whole resolve loop, on every target, against a scripted runtime: no model, no device. */
class FunctionGemmaEngineTest {

    private val catalog = ToolCatalog.parse(
        """{"id":"toy-tools.v1","functions":[
             {"name":"toggle_lamp","description":"Switch the lamp.","parameters":{"state":{"type":"string","description":"on or off","enum":["on","off"]}},"required":["state"]},
             {"name":"roll_dice","description":"Roll a die."}]}""",
    )

    /** One id per "word"; the special markers the engine looks up are single tokens, like in the real tokenizer. */
    private class WordTokenizer : NluTokenizer {
        private val ids = LinkedHashMap<String, Int>().apply { put("<end_of_turn>", FunctionGemmaPrompt.EOS) }
        private val words = HashMap<Int, String>().apply { put(FunctionGemmaPrompt.EOS, "<end_of_turn>") }
        fun id(word: String): Int = ids.getOrPut(word) { (1000 + ids.size).also { words[it] = word } }
        override fun encode(text: String): IntArray =
            Regex("<[a-z_]+>|[^<\\s]+|\\s+").findAll(text).map { id(it.value) }.toList().toIntArray()
        override fun decode(tokens: IntArray): String = tokens.joinToString("") { words.getValue(it) }
    }

    /** Emits a fixed token sequence, one per call, and records how it was driven. */
    private class ScriptedBackend(private val script: IntArray, private val failAtStep: Int = -1) : KvBackend {
        val calls = ArrayList<String>()
        private var cursor = 0
        var closedSnapshots = 0
        private fun next(): Int = script.getOrElse(cursor++) { FunctionGemmaPrompt.EOS }
        override fun prefill(tokens: IntArray, n: Int): Int { calls += "prefill(${tokens.size},$n)"; return 0 }
        override fun releasePrefill() { calls += "releasePrefill" }
        override fun snapshot(): KvBackend.Snapshot { calls += "snapshot"; return object : KvBackend.Snapshot { override fun close() { closedSnapshots++ } } }
        override fun restore(snapshot: KvBackend.Snapshot) { calls += "restore"; cursor = 0 }
        // Every chunk call predicts a next token; only the last one is used. The script starts after the final chunk.
        override fun chunk(tokens: IntArray, n: Int): Int { calls += "chunk(${tokens.size},$n)"; cursor = 0; return next() }
        override fun step(token: Int): Int { if (calls.count { it == "step" } == failAtStep) throw IllegalStateException("device lost"); calls += "step"; return next() }
        override fun close() { calls += "close" }
    }

    private fun engine(tok: WordTokenizer, backend: ScriptedBackend, config: FunctionGemmaEngine.Config = FunctionGemmaEngine.Config(chunk = 4, prefillSeq = 512), cache: PrefixIdCache = PrefixIdCache.None) =
        FunctionGemmaEngine("toy-cartridge", catalog, config, { tok }, { backend }, cache)

    private fun WordTokenizer.script(text: String): IntArray = encode(text)

    @Test
    fun `a closed function call resolves to the catalog function and string arguments`() {
        val tok = WordTokenizer()
        val backend = ScriptedBackend(tok.script("<start_function_call>call:toggle_lamp{state:<escape>off<escape>}<end_function_call><start_function_call>call:roll_dice{}"))
        val e = engine(tok, backend)
        e.warmUp()
        val r = assertIs<NluResolution.Call>(e.resolve("  turn the lamp off  ", budgetMs = 10_000))
        assertEquals("toggle_lamp", r.name)
        assertEquals(mapOf("state" to "off"), r.args)
        assertTrue(r.raw.endsWith("<end_function_call>"), "decoding stops at the first closed call: ${r.raw}")
        assertEquals(setOf("toggle_lamp", "roll_dice"), e.toolNames)
    }

    @Test
    fun `warm-up prefills once and snapshots - resolve restores and feeds padded chunks`() {
        val tok = WordTokenizer()
        val backend = ScriptedBackend(tok.script("<start_function_call>call:roll_dice{}<end_function_call>"))
        val e = engine(tok, backend)
        e.warmUp(); e.warmUp()
        assertEquals(listOf("prefill(512,${e.prefixTokens})", "snapshot", "releasePrefill"), backend.calls.toList())
        assertTrue(e.prefixTokens > 10)

        e.resolve("roll the dice please now", 10_000)
        val afterWarmUp = backend.calls.drop(3)
        assertEquals("restore", afterWarmUp.first())
        val chunks = afterWarmUp.filter { it.startsWith("chunk") }
        assertTrue(chunks.size >= 2 && chunks.all { it.startsWith("chunk(4,") }, "utterance is fed in fixed-size chunks: $chunks")

        e.close()
        assertEquals("close", backend.calls.last())
        assertEquals(1, backend.closedSnapshots)
        assertIs<NluResolution.Failed>(e.resolve("again", 10_000))
    }

    @Test
    fun `a near-miss name is snapped and an unknown one is no call`() {
        val tok = WordTokenizer()
        assertEquals("toggle_lamp", assertIs<NluResolution.Call>(engine(tok, ScriptedBackend(tok.script("<start_function_call>call:Toggle__Lamp{state:<escape>on<escape>}<end_function_call>"))).apply { warmUp() }.resolve("x", 10_000)).name)
        assertIs<NluResolution.NoCall>(engine(tok, ScriptedBackend(tok.script("<start_function_call>call:open_garage{}<end_function_call>"))).apply { warmUp() }.resolve("x", 10_000))
    }

    @Test
    fun `prose is cut short and reported as no call`() {
        val tok = WordTokenizer()
        val backend = ScriptedBackend(tok.script("I am sorry but I cannot help with that request at all today or tomorrow"))
        val e = engine(tok, backend, FunctionGemmaEngine.Config(chunk = 4, prefillSeq = 512, maxProseTokens = 5))
        e.warmUp()
        val r = assertIs<NluResolution.NoCall>(e.resolve("tell me a joke", 10_000))
        assertEquals(5, r.timing.decodeTokens)
    }

    @Test
    fun `an exhausted budget and a runtime failure are reported and never thrown`() {
        val tok = WordTokenizer()
        val slow = engine(tok, ScriptedBackend(tok.script("<start_function_call>call:roll_dice{}<end_function_call>")))
        slow.warmUp()
        assertTrue(assertIs<NluResolution.Failed>(slow.resolve("roll", budgetMs = -1)).reason.contains("budget"))

        val broken = engine(tok, ScriptedBackend(tok.script("<start_function_call>call:roll_dice{}<end_function_call>"), failAtStep = 1))
        broken.warmUp()
        assertTrue(assertIs<NluResolution.Failed>(broken.resolve("roll", 10_000)).reason.contains("device lost"))
    }

    @Test
    fun `a catalog that does not fit the prefill graph is refused with the numbers`() {
        val tok = WordTokenizer()
        val e = engine(tok, ScriptedBackend(IntArray(0)), FunctionGemmaEngine.Config(chunk = 4, prefillSeq = 8))
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
        FunctionGemmaEngine("toy", catalog, FunctionGemmaEngine.Config(4, 512), { counting }, { ScriptedBackend(IntArray(0)) }, cache).warmUp()
        assertEquals(0, counting.prefixEncodes, "a cached prefix must not be re-encoded")
    }
}
