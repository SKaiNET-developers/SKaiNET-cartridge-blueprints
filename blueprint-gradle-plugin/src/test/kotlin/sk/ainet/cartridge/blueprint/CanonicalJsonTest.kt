package sk.ainet.cartridge.blueprint

import kotlinx.serialization.json.Json
import sk.ainet.cartridge.blueprint.core.CanonicalJson
import kotlin.test.Test
import kotlin.test.assertEquals

/** Expected values were produced by CPython: `json.dumps(v)` / `json.dumps(obj, sort_keys=True, separators=(",", ":"))`. */
class CanonicalJsonTest {

    @Test
    fun `floats print like python repr`() {
        val vectors = listOf(
            0.25 to "0.25", 1.0 to "1.0", 5900.0 to "5900.0", 0.1 to "0.1", 1e-05 to "1e-05", 0.0001 to "0.0001",
            0.00012345 to "0.00012345", 1e16 to "1e+16", 1e15 to "1000000000000000.0",
            123456789012345.0 to "123456789012345.0", 1234567890123456.0 to "1234567890123456.0",
            12345678901234567.0 to "1.2345678901234568e+16", 2.5e-7 to "2.5e-07", 1.0 / 3 to "0.3333333333333333",
            100.0 to "100.0", 1e22 to "1e+22", 1.5e300 to "1.5e+300", 5e-324 to "5e-324", -0.0 to "-0.0",
            -2.75 to "-2.75", 0.30000000000000004 to "0.30000000000000004", 9007199254740993.0 to "9007199254740992.0",
            1e21 to "1e+21", 65536.0 to "65536.0", 3.14159 to "3.14159",
        )
        for ((value, expected) in vectors) assertEquals(expected, CanonicalJson.pythonFloatRepr(value), "repr($value)")
    }

    @Test
    fun `object canonicalizes byte-for-byte like python`() {
        // Keys sort by code point (U+FFFF before U+1F600), not by UTF-16 unit; non-ASCII is \u-escaped.
        val source = """{"b":[1,2.50,true,null],"a":"é😀\"\\\n","é":1,"Z":0,"aa":1,"a😀":2,"a￿":3}"""
        val expected = """{"Z":0,"a":"\u00e9\ud83d\ude00\"\\\n","aa":1,"a\uffff":3,"a\ud83d\ude00":2,"b":[1,2.5,true,null],"\u00e9":1}"""
        assertEquals(expected, CanonicalJson.encode(Json.parseToJsonElement(source)))
    }

    @Test
    fun `equivalent number spellings canonicalize identically`() {
        val a = CanonicalJson.encode(Json.parseToJsonElement("""{"rtf":0.250,"n":10}"""))
        val b = CanonicalJson.encode(Json.parseToJsonElement("""{"n":10,"rtf":2.5e-1}"""))
        assertEquals(a, b)
        assertEquals("""{"n":10,"rtf":0.25}""", a)
    }
}
