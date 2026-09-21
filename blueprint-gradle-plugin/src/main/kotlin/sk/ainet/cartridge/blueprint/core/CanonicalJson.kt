package sk.ainet.cartridge.blueprint.core

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.io.File
import java.security.MessageDigest

/**
 * The spec's canonical JSON: what `verify_manifest.py` hashes —
 * `json.dumps(obj, sort_keys=True, separators=(",", ":"))` with Python's default `ensure_ascii=True`.
 * A manifest digest computed here MUST equal the one that script computes, byte for byte.
 */
object CanonicalJson {

    fun encode(element: JsonElement): String = StringBuilder().also { write(element, it) }.toString()

    fun sha256(element: JsonElement): String = Digests.sha256(encode(element).toByteArray(Charsets.US_ASCII))

    private fun write(e: JsonElement, out: StringBuilder) {
        when (e) {
            is JsonNull -> out.append("null")
            is JsonObject -> {
                out.append('{')
                // Python sorts keys by code point; compare by code point, not by UTF-16 unit.
                e.keys.sortedWith(CODE_POINT_ORDER).forEachIndexed { i, k ->
                    if (i > 0) out.append(',')
                    writeString(k, out)
                    out.append(':')
                    write(e.getValue(k), out)
                }
                out.append('}')
            }
            is JsonArray -> {
                out.append('[')
                e.forEachIndexed { i, v ->
                    if (i > 0) out.append(',')
                    write(v, out)
                }
                out.append(']')
            }
            is JsonPrimitive -> if (e.isString) writeString(e.content, out) else writeLiteral(e.content, out)
        }
    }

    /**
     * Numbers and booleans. Integer literals and booleans print identically in Python and here. Any other
     * number is a Python `float`: it is parsed and re-printed the way Python's `repr` does, so that
     * `0.250` and `2.5e-1` canonicalize to the same bytes the reference verifier produces.
     */
    private fun writeLiteral(content: String, out: StringBuilder) {
        if (content == "true" || content == "false" || INTEGER.matches(content)) {
            out.append(content)
            return
        }
        val d = content.toDoubleOrNull()
            ?: throw IllegalArgumentException("Not a JSON number: '$content'")
        require(d.isFinite()) { "Canonical JSON cannot represent $content" }
        out.append(pythonFloatRepr(d))
    }

    /**
     * Python's `repr(float)`: the shortest decimal string that round-trips, in fixed notation when the
     * decimal exponent is in [-4, 16), otherwise as `d.ddde±XX`; fixed notation always carries a fraction.
     * `Double.toString` yields the same shortest digits on JDK 19+ — only the layout differs.
     */
    internal fun pythonFloatRepr(d: Double): String {
        if (d == 0.0) return if (1.0 / d < 0) "-0.0" else "0.0"
        val bd = java.math.BigDecimal(java.lang.Double.toString(Math.abs(d))).stripTrailingZeros()
        var digits = bd.unscaledValue().toString()
        var exp10 = digits.length - bd.scale() - 1          // value = d.ddd × 10^exp10
        // Java always prints at least two significant digits (`4.9E-324`); Python prints the true shortest
        // (`5e-324`). When Java gave two, check whether one already round-trips.
        if (digits.length == 2) {
            val rounded = java.math.BigDecimal(digits).movePointLeft(1).setScale(0, java.math.RoundingMode.HALF_EVEN).toInt()
            val (d1, e1) = if (rounded == 10) 1 to exp10 + 1 else rounded to exp10
            if (d1 in 1..9 && "${d1}e$e1".toDouble() == Math.abs(d)) {
                digits = d1.toString()
                exp10 = e1
            }
        }
        val sign = if (d < 0) "-" else ""
        val body = if (exp10 < -4 || exp10 >= 16) {
            val mantissa = if (digits.length == 1) digits else digits[0] + "." + digits.substring(1)
            val e = Math.abs(exp10).toString().padStart(2, '0')
            mantissa + "e" + (if (exp10 < 0) "-" else "+") + e
        } else if (exp10 < 0) {
            "0." + "0".repeat(-exp10 - 1) + digits
        } else if (digits.length <= exp10 + 1) {
            digits + "0".repeat(exp10 + 1 - digits.length) + ".0"
        } else {
            digits.substring(0, exp10 + 1) + "." + digits.substring(exp10 + 1)
        }
        return sign + body
    }

    private fun writeString(s: String, out: StringBuilder) {
        out.append('"')
        var i = 0
        while (i < s.length) {
            val c = s[i]
            when {
                c == '"' -> out.append("\\\"")
                c == '\\' -> out.append("\\\\")
                c == '\n' -> out.append("\\n")
                c == '\r' -> out.append("\\r")
                c == '\t' -> out.append("\\t")
                c == '\b' -> out.append("\\b")
                c == '\u000C' -> out.append("\\f")
                c.code < 0x20 || c.code > 0x7E -> out.append("\\u").append(c.code.toString(16).padStart(4, '0'))
                else -> out.append(c)
            }
            i++
        }
        out.append('"')
    }

    private val INTEGER = Regex("-?(0|[1-9][0-9]*)")

    private val CODE_POINT_ORDER = Comparator<String> { a, b ->
        val ia = a.codePoints().iterator()
        val ib = b.codePoints().iterator()
        while (ia.hasNext() && ib.hasNext()) {
            val d = ia.nextInt().compareTo(ib.nextInt())
            if (d != 0) return@Comparator d
        }
        ia.hasNext().compareTo(ib.hasNext())
    }
}

object Digests {
    fun sha256(bytes: ByteArray): String =
        "sha256:" + MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 16)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return "sha256:" + md.digest().joinToString("") { "%02x".format(it) }
    }
}
