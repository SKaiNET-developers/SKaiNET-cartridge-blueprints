package sk.ainet.cartridge.blueprint

import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.steps.Int8ParametersRewrite
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class Int8ParametersRewriteTest {
    // @t0: 2 x 8 matrix, loaded + converted (rewritten); @t1: a norm gain (rank 1, stays); @t2: 2 x 8 but used as bf16
    private val module = """
        module {
          util.global private @t0 = #flow.parameter.named<"model"::"t0"> : tensor<2x8xbf16>
          util.global private @t1 = #flow.parameter.named<"model"::"t1"> : tensor<8xbf16>
          util.global private @t2 = #flow.parameter.named<"model"::"t2"> : tensor<2x8xbf16>
          func.func @gemma(%arg0: tensor<1x8xf32>) -> (tensor<1x2xf32>, tensor<2x8xbf16>) {
            %v0_h = util.global.load @t0 : tensor<2x8xbf16>
            %v0 = stablehlo.convert %v0_h : (tensor<2x8xbf16>) -> tensor<2x8xf32>
            %v1_h = util.global.load @t1 : tensor<8xbf16>
            %v2 = util.global.load @t2 : tensor<2x8xbf16>
            %v3 = stablehlo.dot_general %arg0, %v0, contracting_dims = [1] x [1] : (tensor<1x8xf32>, tensor<2x8xf32>) -> tensor<1x2xf32>
            return %v3, %v2 : tensor<1x2xf32>, tensor<2x8xbf16>
          }
        }
    """.trimIndent() + "\n"

    @Test
    fun `a loaded and converted bf16 matrix becomes an int8 and a scale parameter, dequantized under its own name`() {
        val r = Int8ParametersRewrite.rewrite(module, minElements = 16)
        assertEquals(listOf(Int8ParametersRewrite.Matrix("t0", 2, 8)), r.matrices)
        val m = r.mlir
        assertContains(m, "  util.global private @t0_q = #flow.parameter.named<\"model\"::\"t0_q\"> : tensor<2x8xi8>\n")
        assertContains(m, "  util.global private @t0_s = #flow.parameter.named<\"model\"::\"t0_s\"> : tensor<2xf32>\n")
        assertFalse(m.contains("@t0 ="), "the bf16 declaration is gone")
        assertContains(m, "    %v0_q = util.global.load @t0_q : tensor<2x8xi8>\n    %v0_s = util.global.load @t0_s : tensor<2xf32>\n")
        assertContains(m, "%v0_f = stablehlo.convert %v0_q : (tensor<2x8xi8>) -> tensor<2x8xf32>")
        assertContains(m, "%v0_b = stablehlo.broadcast_in_dim %v0_s, dims = [0] : (tensor<2xf32>) -> tensor<2x8xf32>")
        assertContains(m, "    %v0 = stablehlo.multiply %v0_f, %v0_b : tensor<2x8xf32>\n")
        assertContains(m, "@t1 = #flow.parameter.named<\"model\"::\"t1\"> : tensor<8xbf16>", message = "rank 1 stays bf16")
        assertContains(m, "%v2 = util.global.load @t2 : tensor<2x8xbf16>", message = "a matrix not converted to f32 stays")
    }

    @Test
    fun `the safetensors file gets the scales and bytes of the rewritten matrices, the rest copied`() {
        val dir = Files.createTempDirectory("int8-params").toFile()
        // t0: row 0 = -1.0 … 1.0 in bf16 (scale 1/127), row 1 = zeros (scale 1); t1: 8 bf16 values copied
        val t0 = FloatArray(16) { i -> if (i < 8) listOf(-1f, -0.5f, 0f, 0.25f, 0.5f, 0.75f, 1f, 0.125f)[i] else 0f }
        val t1 = FloatArray(8) { it.toFloat() }
        val input = File(dir, "in.safetensors").also { safetensors(it, listOf("t0" to t0, "t1" to t1)) }
        val output = File(dir, "out.safetensors")
        val written = Int8ParametersRewrite.rewriteSafetensors(input, output, listOf(Int8ParametersRewrite.Matrix("t0", 2, 8)))
        assertEquals(output.length(), written)

        val bytes = output.readBytes()
        val n = ByteBuffer.wrap(bytes, 0, 8).order(ByteOrder.LITTLE_ENDIAN).long.toInt()
        assertEquals(0, n % 8, "the data starts 8-byte aligned")
        val header = String(bytes, 8, n)
        assertContains(header, "\"t0_s\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}")
        assertContains(header, "\"t0_q\":{\"dtype\":\"I8\",\"shape\":[16],\"data_offsets\":[8,24]}")
        assertContains(header, "\"t1\":{\"dtype\":\"BF16\",\"shape\":[8],\"data_offsets\":[24,40]}")
        val data = ByteBuffer.wrap(bytes, 8 + n, bytes.size - 8 - n).slice().order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(1f / 127f, data.getFloat(0))
        assertEquals(1f, data.getFloat(4))
        assertEquals(listOf(-127, -64, 0, 32, 64, 95, 127, 16), (0 until 8).map { data.get(8 + it).toInt() })  // half to even: 63.5 → 64, 95.25 → 95, 15.875 → 16
        assertEquals((0 until 8).map { 0 }, (8 until 16).map { data.get(8 + it).toInt() })
        assertEquals(t1.toList(), (0 until 8).map { java.lang.Float.intBitsToFloat((data.getShort(24 + 2 * it).toInt() and 0xFFFF) shl 16) })
    }

    @Test
    fun `a module without a matching parameter is refused`() {
        assertFailsWith<MaterializationException> { Int8ParametersRewrite.rewrite(module.replace("stablehlo.convert %v0_h", "stablehlo.negate %v0_h"), minElements = 16) }
    }

    private fun safetensors(file: File, tensors: List<Pair<String, FloatArray>>) {
        var at = 0
        val header = tensors.joinToString(",", "{", "}") { (k, v) ->
            "\"$k\":{\"dtype\":\"BF16\",\"shape\":[${v.size}],\"data_offsets\":[$at,${(at + 2 * v.size).also { at = it }}]}"
        }.encodeToByteArray()
        val buf = ByteBuffer.allocate(8 + header.size + at).order(ByteOrder.LITTLE_ENDIAN)
        buf.putLong(header.size.toLong()).put(header)
        for ((_, v) in tensors) v.forEach { buf.putShort((java.lang.Float.floatToRawIntBits(it) ushr 16).toShort()) }
        file.writeBytes(buf.array())
    }
}
