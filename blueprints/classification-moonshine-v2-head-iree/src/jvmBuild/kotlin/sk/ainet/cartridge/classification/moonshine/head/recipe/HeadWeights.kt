package sk.ainet.cartridge.classification.moonshine.head.recipe

import kotlinx.io.asSink
import kotlinx.io.buffered
import sk.ainet.cartridge.classification.moonshine.head.model.ClassificationHeadConfig
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsWriter
import sk.ainet.io.safetensors.StreamingSafeTensorsReader
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** The head's parameters by name, as float32 — what `head.safetensors` holds. */
internal class HeadWeights(val tensors: Map<String, Pair<IntArray, FloatArray>>, val metadata: Map<String, String> = emptyMap()) {

    /** Overwrites every parameter of [model]; names and shapes must match exactly, in both directions. */
    fun applyTo(model: Module<FP32, Float>, ctx: ExecutionContext) {
        val parameters = model.trainableParameters()
        val names = parameters.map { it.name }.toSet()
        require(names == tensors.keys) { "head.safetensors and the head disagree: file ${tensors.keys - names}, head ${names - tensors.keys}" }
        for (p in parameters) {
            val (shape, values) = tensors.getValue(p.name)
            val expected = p.value.shape.dimensions
            require(shape.contentEquals(expected)) { "'${p.name}': file ${shape.toList()}, head ${expected.toList()}" }
            @Suppress("UNCHECKED_CAST")
            (p.moduleParameter as ModuleParameter<FP32, Float>).value = ctx.fromFloatArray(Shape(*shape), FP32::class, values)
        }
    }

    fun write(file: File, config: ClassificationHeadConfig) {
        file.absoluteFile.parentFile.mkdirs()
        file.outputStream().asSink().buffered().use { sink ->
            SafeTensorsWriter.write(sink) {
                metadata("model", "classification-head")
                metadata("classes", config.classes.toString())
                metadata("dim", config.dim.toString())
                metadata("max_rows", config.maxRows.toString())
                metadata("hidden", config.hidden.toString())
                for ((k, v) in metadata) metadata(k, v)
                for ((name, t) in tensors.toSortedMap()) tensorF32(name, t.first, t.second)
            }
        }
    }

    companion object {
        fun of(model: Module<FP32, Float>): HeadWeights = HeadWeights(
            model.trainableParameters().associate { p -> p.name to (p.value.shape.dimensions.copyOf() to floats(p.value.data)) },
        )

        fun read(file: File): HeadWeights = StreamingSafeTensorsReader.open(JvmRandomAccessSource.open(file)).use { reader ->
            HeadWeights(
                reader.tensors.associate { t ->
                    require(t.dtype == "F32") { "${file.name}: '${t.name}' is ${t.dtype}, expected F32" }
                    t.name to (t.shape.map(Long::toInt).toIntArray() to littleEndianFloats(reader.loadTensorData(t)))
                },
                reader.metadata.toMap(),
            )
        }

        fun config(file: File, dropout: Float = 0f): ClassificationHeadConfig {
            val m = read(file).metadata
            fun int(key: String) = m[key]?.toInt() ?: error("${file.name}: no '$key' in the metadata")
            return ClassificationHeadConfig(classes = int("classes"), dim = int("dim"), maxRows = int("max_rows"), hidden = int("hidden"), dropout = dropout)
        }

        private fun floats(data: sk.ainet.lang.tensor.data.TensorData<*, *>): FloatArray {
            val shape = data.shape.dimensions
            val n = shape.fold(1) { a, b -> a * b }
            val out = FloatArray(n)
            val index = IntArray(shape.size)
            for (i in 0 until n) {
                var rest = i
                for (d in shape.indices.reversed()) { index[d] = rest % shape[d]; rest /= shape[d] }
                out[i] = (data.get(*index) as Number).toFloat()
            }
            return out
        }

        internal fun littleEndianFloats(bytes: ByteArray): FloatArray =
            FloatArray(bytes.size / 4).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
    }
}
