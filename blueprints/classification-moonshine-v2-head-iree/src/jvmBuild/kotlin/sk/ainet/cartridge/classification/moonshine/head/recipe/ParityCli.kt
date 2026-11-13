package sk.ainet.cartridge.classification.moonshine.head.recipe

import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import sk.ainet.cartridge.classification.moonshine.head.Decision
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * Verification `host-parity`: the compiled module against the eager head, on real test rows. Two phases around
 * one `iree-run-module` call per input (the `parityHost` task runs all three):
 *
 *   --mode write   --weights <head.safetensors> --features <dir> --label-set <label-set.json> --dir <work> [--count 8]
 *   --mode compare --dir <work> [--tolerance 0.001]
 *
 * `write` leaves `input-<i>.npy` (`[1, max_rows, dim + 1]`) and the eager logits; `compare` reads `output-<i>.npy`.
 */
public fun main(args: Array<String>) {
    val a = Args(args)
    val dir = a.file("dir").apply { mkdirs() }
    val expectedFile = File(dir, "expected.json")
    val rows = ListSerializer(ListSerializer(Float.serializer()))
    when (val mode = a.string("mode")) {
        "write" -> {
            val weights = a.file("weights")
            val config = HeadWeights.config(weights)
            val test = FeatureStore(a.file("features")).examples("test", LabelSetInput.read(a.file("label-set"))).take(a.int("count", 8))
            require(test.isNotEmpty()) { "no `test` recordings to compare on" }
            // One at a time and padded to max_rows, exactly as the compiled function is fed.
            val head = EagerHead.of(weights)
            val expected = test.mapIndexed { i, e ->
                Npy.write(File(dir, "input-$i.npy"), intArrayOf(1, config.maxRows, config.rowWidth), pack(listOf(e), config.dim, config.maxRows))
                head.logits(listOf(e)).single().toList()
            }
            expectedFile.writeText(json.encodeToString(rows, expected))
            println("parity: ${test.size} inputs in $dir")
        }
        "compare" -> {
            val tolerance = a.float("tolerance", 0.001f)
            val expected = json.decodeFromString(rows, expectedFile.readText())
            var worst = 0f
            for ((i, e) in expected.withIndex()) {
                val got = Npy.read(File(dir, "output-$i.npy"))
                check(got.size == e.size) { "input $i: ${got.size} logits from the module, ${e.size} from the eager head" }
                for (c in e.indices) worst = maxOf(worst, abs(got[c] - e[c]))
                check(Decision.argmax(got) == Decision.argmax(e.toFloatArray())) { "input $i: the compiled module and the eager head disagree on the top class" }
            }
            println("parity: ${expected.size} inputs, max |difference| $worst (tolerance $tolerance)")
            check(worst <= tolerance) { "host parity failed: max |difference| $worst exceeds $tolerance" }
        }
        else -> error("--mode must be write or compare, got '$mode'")
    }
}

/** NumPy `.npy` files of little-endian float32, the format `iree-run-module` reads and writes. */
internal object Npy {
    fun write(file: File, shape: IntArray, values: FloatArray) {
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.joinToString(", ")}${if (shape.size == 1) "," else ""}), }"
        header += " ".repeat((64 - (10 + header.length + 1) % 64) % 64) + "\n"
        val out = ByteBuffer.allocate(10 + header.length + values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        out.put(byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte(), 1, 0))
        out.putShort(header.length.toShort()).put(header.toByteArray(Charsets.US_ASCII))
        for (v in values) out.putFloat(v)
        file.writeBytes(out.array())
    }

    fun read(file: File): FloatArray {
        val bytes = file.readBytes()
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val major = bytes[6].toInt()
        val headerLength = if (major == 1) b.getShort(8).toInt() and 0xffff else b.getInt(8)
        val start = (if (major == 1) 10 else 12) + headerLength
        val header = String(bytes, 0, start, Charsets.ISO_8859_1)
        check("'<f4'" in header) { "${file.name}: expected float32, header $header" }
        return FloatArray((bytes.size - start) / 4) { b.getFloat(start + it * 4) }
    }
}
