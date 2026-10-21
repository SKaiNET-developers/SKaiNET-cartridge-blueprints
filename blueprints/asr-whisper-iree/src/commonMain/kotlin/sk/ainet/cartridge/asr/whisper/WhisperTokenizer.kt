package sk.ainet.cartridge.asr.whisper

import kotlinx.io.files.Path
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Token ids → text with the checkpoint's `tokenizer.json` (Hugging Face, byte-level BPE): the decode half only — the
 * runtime produces ids, it never encodes. Special and added tokens (`<|startoftranscript|>`, language, task,
 * timestamp tokens, end-of-text) decode to nothing.
 */
public class WhisperTokenizer private constructor(
    /** id → byte-level piece; `null` for ids that decode to nothing. */
    private val pieces: Array<String?>,
) {
    /** The vocabulary size the file covers (highest id + 1). */
    public val size: Int get() = pieces.size

    /** The text of [ids], special tokens skipped, as UTF-8 (a character split across tokens is rejoined). */
    public fun decode(ids: IntArray): String {
        val bytes = ArrayList<Byte>(ids.size * 4)
        for (id in ids) {
            val piece = pieces.getOrNull(id) ?: continue
            for (ch in piece) {
                val b = UNICODE_TO_BYTE[ch]
                if (b != null) bytes += b else bytes.addAll(ch.toString().encodeToByteArray().toList())
            }
        }
        return bytes.toByteArray().decodeToString()
    }

    public companion object {
        private val json = Json { ignoreUnknownKeys = true }

        /** Reads `tokenizer.json`'s `model.vocab` and `added_tokens`. */
        public fun fromJson(text: String): WhisperTokenizer {
            val root = json.parseToJsonElement(text).jsonObject
            val vocab = root.getValue("model").jsonObject.getValue("vocab").jsonObject
            val added = root["added_tokens"]?.jsonArray.orEmpty().map { it.jsonObject }
            val maxId = maxOf(
                vocab.values.maxOfOrNull { it.jsonPrimitive.int } ?: -1,
                added.maxOfOrNull { it.getValue("id").jsonPrimitive.int } ?: -1,
            )
            val pieces = arrayOfNulls<String>(maxId + 1)
            for ((piece, id) in vocab) pieces[id.jsonPrimitive.int] = piece
            // added tokens are the specials and timestamps: none of them is transcript text
            for (t in added) {
                val special = t["special"]?.jsonPrimitive?.boolean ?: true
                val content = t.getValue("content").jsonPrimitive.content
                if (special || (content.startsWith("<|") && content.endsWith("|>"))) pieces[t.getValue("id").jsonPrimitive.int] = null
            }
            return WhisperTokenizer(pieces)
        }

        public fun fromFile(path: Path): WhisperTokenizer = fromJson(PackDir.readText(path))

        /**
         * GPT-2's byte-level alphabet, reversed: printable Latin-1 bytes stand for themselves, the other 68 bytes for
         * the code points from U+0100 on, in byte order.
         */
        private val UNICODE_TO_BYTE: Map<Char, Byte> = buildMap {
            val printable = (0x21..0x7E) + (0xA1..0xAC) + (0xAE..0xFF)
            var next = 0x100
            for (b in 0..0xFF) {
                val ch = if (b in printable) b.toChar() else (next++).toChar()
                put(ch, b.toByte())
            }
        }
    }
}
