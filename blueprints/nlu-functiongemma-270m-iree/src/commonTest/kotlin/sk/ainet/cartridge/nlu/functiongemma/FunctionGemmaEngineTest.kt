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
             {"name":"toggle_lamp","description":"Switch the lamp.","parameters":{
                "state":{"type":"string","description":"on or off","enum":["on","off"]},
                "level":{"type":"string","description":"brightness"}},"required":["state"]},
             {"name":"roll_dice","description":"Roll a die."}]}""",
    )

    /** Words, numbers and single punctuation marks, like the real tokenizer splits a call; special markers are single tokens. */
    private class WordTokenizer : NluTokenizer {
        private val ids = LinkedHashMap<String, Int>().apply { put("<end_of_turn>", FunctionGemmaPrompt.EOS) }
        private val words = HashMap<Int, String>().apply { put(FunctionGemmaPrompt.EOS, "<end_of_turn>") }
        fun id(word: String): Int = ids.getOrPut(word) { (1000 + ids.size).also { words[it] = word } }
        override fun encode(text: String): IntArray =
            Regex("<[a-z_]+>|[A-Za-z0-9]+|\\s+|[^A-Za-z0-9<\\s]").findAll(text).map { id(it.value) }.toList().toIntArray()
        override fun decode(tokens: IntArray): String = tokens.joinToString("") { words.getValue(it) }
    }

    /**
     * A model that knows its answer: once the utterance is in, every call predicts the [script] token at the
     * number of answer tokens it has been fed so far, whatever they were. Records how it was driven.
     */
    private class ScriptedBackend(private val script: IntArray, private val failAtCall: Int = -1) : KvBackend {
        val calls = ArrayList<String>()
        var utteranceTokens = 0
        private var fed = 0
        var closedSnapshots = 0
        private fun next(): Int = script.getOrElse(fed - utteranceTokens) { FunctionGemmaPrompt.EOS }
        private fun call(what: String) {
            if (failAtCall >= 0 && calls.count { it == "step" || it.startsWith("chunk") } == failAtCall) throw IllegalStateException("device lost")
            calls += what
        }
        override fun prefill(tokens: IntArray, n: Int): Int { calls += "prefill(${tokens.size},$n)"; return 0 }
        override fun releasePrefill() { calls += "releasePrefill" }
        override fun snapshot(): KvBackend.Snapshot { calls += "snapshot"; return object : KvBackend.Snapshot { override fun close() { closedSnapshots++ } } }
        override fun restore(snapshot: KvBackend.Snapshot) { calls += "restore"; fed = 0 }
        override fun chunk(tokens: IntArray, n: Int): Int { call("chunk(${tokens.size},$n)"); fed += n; return next() }
        override fun step(token: Int): Int { call("step"); fed++; return next() }
        override fun close() { calls += "close" }
    }

    private fun engine(tok: WordTokenizer, backend: ScriptedBackend, config: FunctionGemmaEngine.Config = FunctionGemmaEngine.Config(chunk = 4, prefillSeq = 512), cache: PrefixIdCache = PrefixIdCache.None) =
        FunctionGemmaEngine("toy-cartridge", catalog, config, { tok }, { backend }, cache)

    private fun WordTokenizer.script(text: String): IntArray = encode(text)

    /** Resolves [text], telling the scripted model where the utterance ends. */
    private fun FunctionGemmaEngine.resolve(tok: WordTokenizer, backend: ScriptedBackend, text: String, budgetMs: Long = 10_000): NluResolution {
        backend.utteranceTokens = tok.encode(FunctionGemmaPrompt(catalog.definitions()).utteranceText(text.trim())).size
        return resolve(text, budgetMs)
    }

    @Test
    fun `a closed function call resolves to the catalog function and string arguments`() {
        val tok = WordTokenizer()
        val backend = ScriptedBackend(tok.script("<start_function_call>call:toggle_lamp{state:<escape>off<escape>}<end_function_call><start_function_call>call:roll_dice{}"))
        val e = engine(tok, backend)
        e.warmUp()
        val r = assertIs<NluResolution.Call>(e.resolve(tok, backend, "  turn the lamp off  "))
        assertEquals("toggle_lamp", r.name)
        assertEquals(mapOf("state" to "off"), r.args)
        assertTrue(r.raw.endsWith("}<end_function_call>"), "decoding stops at the first closed call: ${r.raw}")
        assertEquals(setOf("toggle_lamp", "roll_dice"), e.toolNames)
    }

    @Test
    fun `forced tokens go to the model in chunk calls - same call in fewer calls`() {
        val answer = "<start_function_call>call:toggle_lamp{level:<escape>dim<escape>,state:<escape>on<escape>}<end_function_call>"
        fun run(config: FunctionGemmaEngine.Config): Pair<NluResolution.Call, ScriptedBackend> {
            val tok = WordTokenizer()
            val backend = ScriptedBackend(tok.script(answer))
            val e = engine(tok, backend, config).apply { warmUp() }
            return assertIs<NluResolution.Call>(e.resolve(tok, backend, "lamp on, dim")) to backend
        }
        val (small, b4) = run(FunctionGemmaEngine.Config(chunk = 4, prefillSeq = 512))
        val (stepped, b32) = run(FunctionGemmaEngine.Config(chunk = 32, prefillSeq = 512))
        assertEquals(mapOf("level" to "dim", "state" to "on"), small.args)
        assertEquals(small.args, stepped.args)
        assertEquals(small.raw, stepped.raw)
        assertEquals(answer, small.raw)
        // 20 answer tokens; the model chooses 8 (start, toggle, level, <escape>, dim, <escape>, `,`, on): `state` is the
        // only key left after the comma, `}` the only way on after the last parameter
        assertEquals(20, small.timing.decodeTokens)
        assertEquals(7, small.timing.decodeCalls, "chunk 4: ${b4.calls}")
        // chunk 32: forced tokens are stepped; the last run (on, <escape>, `}`) is never fed
        assertEquals(16, stepped.timing.decodeCalls)
        assertTrue(b32.calls.dropWhile { it != "step" }.all { it == "step" }, "${b32.calls}")
    }

    @Test
    fun `an answer the grammar does not expect is decoded as the model says`() {
        val tok = WordTokenizer()
        val answer = "<start_function_call>call:toggle_lamp{colour:<escape>red<escape>}<end_function_call>"
        val backend = ScriptedBackend(tok.script(answer))
        val e = engine(tok, backend).apply { warmUp() }
        val r = assertIs<NluResolution.Call>(e.resolve(tok, backend, "red lamp"))
        assertEquals(answer, r.raw)
        assertEquals(mapOf("colour" to "red"), r.args)
    }

    @Test
    fun `warm-up prefills once and snapshots - resolve restores and feeds padded chunks`() {
        val tok = WordTokenizer()
        val backend = ScriptedBackend(tok.script("<start_function_call>call:roll_dice{}<end_function_call>"))
        val e = engine(tok, backend)
        e.warmUp(); e.warmUp()
        assertEquals(listOf("prefill(512,${e.prefixTokens})", "snapshot", "releasePrefill"), backend.calls.toList())
        assertTrue(e.prefixTokens > 10)

        val r = assertIs<NluResolution.Call>(e.resolve(tok, backend, "roll the dice please now"))
        assertEquals("roll_dice", r.name)
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
        val near = ScriptedBackend(tok.script("<start_function_call>call:Toggle__Lamp{state:<escape>on<escape>}<end_function_call>"))
        assertEquals("toggle_lamp", assertIs<NluResolution.Call>(engine(tok, near).apply { warmUp() }.resolve(tok, near, "x")).name)
        val unknown = ScriptedBackend(tok.script("<start_function_call>call:open_garage{}<end_function_call>"))
        assertIs<NluResolution.NoCall>(engine(tok, unknown).apply { warmUp() }.resolve(tok, unknown, "x"))
    }

    @Test
    fun `prose is cut short and reported as no call`() {
        val tok = WordTokenizer()
        val backend = ScriptedBackend(tok.script("I am sorry but I cannot help with that request at all today or tomorrow"))
        val e = engine(tok, backend, FunctionGemmaEngine.Config(chunk = 4, prefillSeq = 512, maxProseTokens = 5))
        e.warmUp()
        val r = assertIs<NluResolution.NoCall>(e.resolve(tok, backend, "tell me a joke"))
        assertEquals(5, r.timing.decodeTokens)
    }

    @Test
    fun `an exhausted budget and a runtime failure are reported and never thrown`() {
        val tok = WordTokenizer()
        val answer = "<start_function_call>call:toggle_lamp{state:<escape>on<escape>}<end_function_call>"
        val slowBackend = ScriptedBackend(tok.script(answer))
        val slow = engine(tok, slowBackend)
        slow.warmUp()
        assertTrue(assertIs<NluResolution.Failed>(slow.resolve(tok, slowBackend, "lamp", budgetMs = -1)).reason.contains("budget"))

        val brokenBackend = ScriptedBackend(tok.script(answer), failAtCall = 1)
        val broken = engine(tok, brokenBackend)
        broken.warmUp()
        assertTrue(assertIs<NluResolution.Failed>(broken.resolve(tok, brokenBackend, "lamp")).reason.contains("device lost"))
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
