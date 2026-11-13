package sk.ainet.cartridge.classification.moonshine.head.recipe

import sk.ainet.cartridge.classification.moonshine.head.model.ClassificationHeadConfig
import sk.ainet.cartridge.classification.moonshine.head.model.classificationHead
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.random.Random

/** Rows and mask of [examples] as one `[batch, rows, dim + 1]` block, padded to [rows] rows. */
internal fun pack(examples: List<Example>, dim: Int, rows: Int): FloatArray {
    val width = dim + 1
    val x = FloatArray(examples.size * rows * width)
    for ((i, e) in examples.withIndex()) {
        for (t in 0 until minOf(e.count, rows)) {
            val at = (i * rows + t) * width
            e.rows.copyInto(x, at, t * dim, (t + 1) * dim)
            x[at + dim] = 1f
        }
    }
    return x
}

/** The second execution of the head: eager, on the CPU. What dev accuracy, calibration and the report are computed with. */
internal class EagerHead(private val config: ClassificationHeadConfig, private val model: Module<FP32, Float>, private val cpu: DirectCpuExecutionContext) {

    /** Logits per example, `[classes]` each. Batches are padded to their own longest example: padding never matters. */
    fun logits(examples: List<Example>, batch: Int = 128): List<FloatArray> = examples.chunked(batch).flatMap { chunk ->
        val rows = chunk.maxOf { it.count }
        val out = model.forward(cpu.fromFloatArray(Shape(chunk.size, rows, config.rowWidth), FP32::class, pack(chunk, config.dim, rows)), cpu)
        chunk.indices.map { i -> FloatArray(config.classes) { c -> out.data.get(i, c) } }
    }

    companion object {
        fun of(weights: File): EagerHead {
            val cpu = DirectCpuExecutionContext()
            val config = HeadWeights.config(weights)
            val model = classificationHead(config, cpu, Random(0))
            HeadWeights.read(weights).applyTo(model, cpu)
            return EagerHead(config, model, cpu)
        }
    }
}
