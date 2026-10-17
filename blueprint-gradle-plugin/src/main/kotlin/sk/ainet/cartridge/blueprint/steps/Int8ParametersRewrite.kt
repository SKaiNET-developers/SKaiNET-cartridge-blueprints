package sk.ainet.cartridge.blueprint.steps

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import kotlinx.serialization.json.put
import sk.ainet.cartridge.blueprint.model.MaterializationException
import java.io.File
import java.io.OutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * A `convert` step on an exported StableHLO module with external parameters: weight-only int8 for the large bf16
 * matrices. The parameter counterpart of [Int8WeightsRewrite], for exporters that keep the weights out of the graph.
 *
 * In the module, every rank-2 bf16 parameter with at least [DEFAULT_MIN_ELEMENTS] elements whose every use is
 * `%x_h = util.global.load @g` + `%x = stablehlo.convert %x_h` (bf16 → f32) becomes two parameters, `<key>_q` (i8)
 * and `<key>_s` (one f32 scale per row), dequantized in the graph under the name `%x`; downstream uses are unchanged.
 * In the safetensors file, the bf16 entry is replaced by those two (`scale = max|row| / 127`, round half to even,
 * clamp to ±127); every other entry is copied as is. Rank-1 parameters (norm gains) stay bf16.
 *
 * Compile the rewritten module with [COMPILE_FLAGS]: with const-expr hoisting on, the compiler would hoist the
 * dequantization into an initializer and keep f32 copies of the weights.
 */
object Int8ParametersRewrite {
    const val DEFAULT_MIN_ELEMENTS = 16384
    val COMPILE_FLAGS = Int8WeightsRewrite.COMPILE_FLAGS

    private val DECL = Regex("""(?m)^( *)util\.global private @(\w+) = #flow\.parameter\.named<"([^"]*)"::"([^"]*)"> : tensor<(\d+)x(\d+)xbf16>\n""")
    private val LOAD_CONVERT = Regex(
        """(?m)^( *)(%\w+)_h = util\.global\.load @(\w+) : tensor<(\d+)x(\d+)xbf16>\n *\2 = stablehlo\.convert \2_h : """ +
            """\(tensor<\4x\5xbf16>\) -> tensor<\4x\5xf32>\n""",
    )
    private val ANY_LOAD = Regex("""util\.global\.load @(\w+) :""")
    private val SUFFIXED = Regex("""%\w+_[qsfb]\b""")

    /** A rewritten parameter: its archive key and matrix shape. */
    data class Matrix(val key: String, val rows: Int, val cols: Int)

    data class Result(val mlir: String, val matrices: List<Matrix>)

    fun rewrite(mlir: String, minElements: Int = DEFAULT_MIN_ELEMENTS): Result {
        SUFFIXED.find(mlir)?.let { throw MaterializationException("int8-parameters: the module already has a value named ${it.value}") }
        val loads = ANY_LOAD.findAll(mlir).groupingBy { it.groupValues[1] }.eachCount()
        val pairs = LOAD_CONVERT.findAll(mlir).groupingBy { it.groupValues[3] }.eachCount()
        val chosen = LinkedHashMap<String, Matrix>()  // global → matrix
        for (d in DECL.findAll(mlir)) {
            val (_, global, _, key, rows, cols) = d.destructured
            if (rows.toLong() * cols.toLong() < minElements) continue
            val n = loads[global] ?: continue
            if (pairs[global] == n) chosen[global] = Matrix(key, rows.toInt(), cols.toInt())
        }
        if (chosen.isEmpty()) throw MaterializationException(
            "int8-parameters: no rank-2 bf16 parameter with at least $minElements elements, loaded and converted to f32, " +
                "found — the module has none, or the exporter's output format changed.",
        )
        val declared = DECL.replace(mlir) { d ->
            val (ind, global, scope, key, rows, cols) = d.destructured
            if (global !in chosen) d.value
            else "${ind}util.global private @${global}_q = #flow.parameter.named<\"$scope\"::\"${key}_q\"> : tensor<${rows}x${cols}xi8>\n" +
                "${ind}util.global private @${global}_s = #flow.parameter.named<\"$scope\"::\"${key}_s\"> : tensor<${rows}xf32>\n"
        }
        val loaded = LOAD_CONVERT.replace(declared) { l ->
            val (ind, v, global, rows, cols) = l.destructured
            if (global !in chosen) return@replace l.value
            val m = "${rows}x$cols"
            "$ind${v}_q = util.global.load @${global}_q : tensor<${m}xi8>\n" +
                "$ind${v}_s = util.global.load @${global}_s : tensor<${rows}xf32>\n" +
                "$ind${v}_f = stablehlo.convert ${v}_q : (tensor<${m}xi8>) -> tensor<${m}xf32>\n" +
                "$ind${v}_b = stablehlo.broadcast_in_dim ${v}_s, dims = [0] : (tensor<${rows}xf32>) -> tensor<${m}xf32>\n" +
                "$ind$v = stablehlo.multiply ${v}_f, ${v}_b : tensor<${m}xf32>\n"
        }
        return Result(loaded, chosen.values.toList())
    }

    /**
     * Writes [output]: the safetensors [input] with each of [matrices] (bf16) replaced by `<key>_q` (I8) and
     * `<key>_s` (F32 per-row scales). Returns the number of bytes written.
     */
    fun rewriteSafetensors(input: File, output: File, matrices: List<Matrix>): Long {
        val byKey = matrices.associateBy { it.key }
        RandomAccessFile(input, "r").use { raf ->
            val ch = raf.channel
            val lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
            ch.read(lenBuf, 0)
            val headerLen = lenBuf.getLong(0)
            val headerBytes = ByteBuffer.allocate(headerLen.toInt()).also { ch.read(it, 8) }.array()
            val header = Json.parseToJsonElement(headerBytes.decodeToString()).jsonObject
            val dataStart = 8 + headerLen

            data class Entry(val key: String, val dtype: String, val shape: List<Long>, val from: Long, val to: Long)
            val entries = header.filterKeys { it != "__metadata__" }.map { (k, v) ->
                val o = v.jsonObject
                val off = o.getValue("data_offsets").jsonArray
                Entry(k, o.getValue("dtype").jsonPrimitive.content, o.getValue("shape").jsonArray.map { it.jsonPrimitive.long },
                    off[0].jsonPrimitive.long, off[1].jsonPrimitive.long)
            }.sortedBy { it.from }
            val missing = byKey.keys - entries.map { it.key }.toSet()
            if (missing.isNotEmpty()) throw MaterializationException("int8-parameters: ${input.name} has no entry ${missing.sorted()}")

            // the new layout: per quantized entry the scales first (keeps the f32 data 4-byte aligned), then the bytes
            val layout = ArrayList<Pair<String, JsonObject>>()
            var at = 0L
            fun place(key: String, dtype: String, shape: List<Long>, bytes: Long) {
                layout += key to buildJsonObject {
                    put("dtype", dtype)
                    put("shape", JsonArray(shape.map(::JsonPrimitive)))
                    put("data_offsets", JsonArray(listOf(JsonPrimitive(at), JsonPrimitive(at + bytes))))
                }
                at += bytes
            }
            for (e in entries) {
                val m = byKey[e.key]
                if (m == null) { place(e.key, e.dtype, e.shape, e.to - e.from); continue }
                val n = m.rows.toLong() * m.cols
                if (e.dtype != "BF16" || e.to - e.from != n * 2) throw MaterializationException(
                    "int8-parameters: ${e.key} is ${e.dtype}, ${e.to - e.from} bytes; the module declares ${m.rows}x${m.cols} bf16",
                )
                place("${e.key}_s", "F32", listOf(m.rows.toLong()), m.rows * 4L)
                place("${e.key}_q", "I8", listOf(n), n)
            }
            val meta = header["__metadata__"]
            var json = Json.encodeToString(JsonObject.serializer(), JsonObject(buildMap {
                if (meta != null) put("__metadata__", meta)
                layout.forEach { (k, v) -> put(k, v) }
            })).encodeToByteArray()
            json += ByteArray((8 - json.size % 8) % 8) { ' '.code.toByte() }  // the data starts 8-byte aligned

            output.parentFile.mkdirs()
            output.outputStream().buffered(1 shl 20).use { out ->
                out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(json.size.toLong()).array())
                out.write(json)
                for (e in entries) {
                    val m = byKey[e.key]
                    if (m == null) { copy(ch, dataStart + e.from, e.to - e.from, out); continue }
                    quantize(ch.map(FileChannel.MapMode.READ_ONLY, dataStart + e.from, e.to - e.from).order(ByteOrder.LITTLE_ENDIAN), m, out)
                }
            }
            return 8L + json.size + at
        }
    }

    private fun copy(ch: FileChannel, from: Long, length: Long, out: OutputStream) {
        val buf = ByteBuffer.allocate(1 shl 20)
        var pos = from
        val end = from + length
        while (pos < end) {
            buf.clear().limit(minOf(buf.capacity().toLong(), end - pos).toInt())
            val n = ch.read(buf, pos)
            if (n <= 0) throw MaterializationException("int8-parameters: short read at $pos")
            out.write(buf.array(), 0, n)
            pos += n
        }
    }

    /** Per-row symmetric int8 of a bf16 matrix: the scales, then the bytes, as [rewrite]'s graph reads them. */
    private fun quantize(src: ByteBuffer, m: Matrix, out: OutputStream) {
        val row = FloatArray(m.cols)
        val scales = ByteBuffer.allocate(m.rows * 4).order(ByteOrder.LITTLE_ENDIAN)
        val q = ByteArray(m.rows * m.cols)
        for (r in 0 until m.rows) {
            var max = 0f
            for (c in 0 until m.cols) {
                val bits = src.getShort((r * m.cols + c) * 2).toInt() and 0xFFFF
                row[c] = java.lang.Float.intBitsToFloat(bits shl 16)
                max = maxOf(max, Math.abs(row[c]))
            }
            val s = if (max == 0f) 1f else max / 127f
            scales.putFloat(s)
            for (c in 0 until m.cols) q[r * m.cols + c] = Math.rint((row[c] / s).toDouble()).coerceIn(-127.0, 127.0).toInt().toByte()
        }
        out.write(scales.array())
        out.write(q)
    }
}
