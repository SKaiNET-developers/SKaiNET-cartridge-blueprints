package sk.ainet.cartridge.asr.whisper

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals

class WhisperTokenizerTest {
    // byte-level pieces: `Ġ` is the space byte, `Ã¼` the two UTF-8 bytes of `ü`; ids 5+ are added tokens
    private val fixture = """
        {"model": {"type": "BPE", "vocab": {"Hallo": 0, "Ġwelt": 1, ".": 2, "Ġgr": 3, "Ã¼": 4}},
         "added_tokens": [
           {"id": 5, "content": "<|endoftext|>", "special": true},
           {"id": 6, "content": "<|de|>", "special": true},
           {"id": 7, "content": "<|0.00|>", "special": false}
         ]}
    """.trimIndent()

    @Test
    fun byteLevelPiecesDecodeToUtf8() {
        val t = WhisperTokenizer.fromJson(fixture)
        assertEquals(8, t.size)
        assertEquals("Hallo welt.", t.decode(intArrayOf(0, 1, 2)))
        assertEquals(" grü", t.decode(intArrayOf(3, 4)))
    }

    @Test
    fun specialAndTimestampTokensDecodeToNothing() {
        val t = WhisperTokenizer.fromJson(fixture)
        assertEquals("Hallo.", t.decode(intArrayOf(6, 7, 0, 2, 5)))
        assertEquals("", t.decode(intArrayOf(99)))
    }

    @Test
    fun theGermanCheckpointDecodesTheReferenceTranscript() {
        // tokenizer.json of the de flavor, once blueprintFetch has run; the ids are the reference greedy decode of the
        // German 4 s clip SKaiNET-transformers' Whisper tests use
        val file = File(System.getProperty("blueprint.dir"), "build/blueprint/sources/weights-de/tokenizer.json")
        if (!file.isFile) { println("SKIP: no fetched weights-de/tokenizer.json"); return }
        val t = WhisperTokenizer.fromJson(file.readText())
        assertEquals("Fern sie ihr lauter.", t.decode(intArrayOf(50258, 50261, 50359, 50363, 37, 1248, 2804, 5553, 635, 20314, 13, 50257)))
    }
}
