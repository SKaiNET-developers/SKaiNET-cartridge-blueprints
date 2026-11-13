package sk.ainet.cartridge.classification.moonshine.head.model

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.layers.Dropout
import sk.ainet.lang.nn.topology.MLP
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.narrow
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.reshape
import sk.ainet.lang.tensor.softmax
import sk.ainet.lang.tensor.sum
import sk.ainet.lang.tensor.times
import sk.ainet.lang.tensor.minus
import sk.ainet.lang.types.FP32
import kotlin.random.Random

/**
 * Geometry of the head. [dim] is the encoder's row width and [maxRows] the number of rows the graph takes; both are
 * the encoder's facts (`attributes.features` in the descriptor), [classes] is the label set's size.
 */
public data class ClassificationHeadConfig(
    val classes: Int,
    val dim: Int = 320,
    val maxRows: Int = 256,
    val hidden: Int = 256,
    val dropout: Float = 0.1f,
) {
    /** One input row: [dim] features and, last, the validity mask (1 for an encoder row, 0 for padding). */
    val rowWidth: Int get() = dim + 1

    init {
        require(classes >= 2) { "a classifier needs at least two classes, got $classes" }
        require(dim > 0 && maxRows > 0 && hidden > 0) { "dim, maxRows and hidden must be positive" }
        require(dropout in 0f..0.9f) { "dropout must be in [0, 0.9], got $dropout" }
    }
}

/**
 * Attention pooling over the valid rows: one learned score per row, a softmax over the rows that are there, and
 * the weighted sum of the rows.
 *
 * Input `[batch, rows, dim + 1]` — the last column is the mask. Output `[batch, dim]`. A padded row gets
 * `score − 1e9` before the softmax, so its weight is exactly zero in float32 whatever the row holds; there is no
 * `where`, only arithmetic that every backend and the exporter share.
 *
 * The score reads the whole input row, mask column included: on a real row that column is 1, so its weight is
 * one more bias, and a padded row's score does not matter.
 */
public class MaskedAttentionPool(
    private val dim: Int,
    ctx: ExecutionContext,
    random: Random,
    override val name: String = "pool",
) : Module<FP32, Float>(), ModuleParameters<FP32, Float> {

    private val width = dim + 1

    private val score = Linear<FP32, Float>(
        inFeatures = width,
        outFeatures = 1,
        name = "$name.score",
        initWeights = ctx.fromFloatArray(Shape(1, width), FP32::class, FloatArray(width) { (random.nextFloat() - 0.5f) * 0.02f }),
        initBias = ctx.fromFloatArray(Shape(1), FP32::class, floatArrayOf(0f)),
    )

    override val modules: List<Module<FP32, Float>> get() = listOf(score)
    override val params: List<ModuleParameter<FP32, Float>> get() = emptyList()

    override fun onForward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val (batch, rows) = input.shape.dimensions
        val mask = input.narrow(dim = 2, start = dim, length = 1)                       // [b, t, 1]
        val scores = score.forward(input, ctx) + (mask - 1f) * MASKED                   // [b, t, 1]
        val weights = scores.reshape(Shape(batch, rows)).softmax(dim = 1)               // over the rows
        // Every row times its weight, as a product of two equal shapes: the weights are first spread across the
        // row width by a matmul with a row of ones. Same numbers as a broadcast, on every backend's fast path.
        val ones = ctx.fromFloatArray<FP32, Float>(Shape(1, width), FP32::class, FloatArray(width) { 1f })
        val spread = ctx.ops.matmul(weights.reshape(Shape(batch * rows, 1)), ones)      // [b·t, width]
        val weighted = input.reshape(Shape(batch * rows, width)) * spread
        return weighted.reshape(Shape(batch, rows, width)).sum(dim = 1)                 // [b, width]
            .narrow(dim = 1, start = 0, length = dim)                                   // the mask column goes
    }

    private companion object {
        const val MASKED = 1e9f
    }
}

/**
 * The head: masked attention pooling, one hidden layer, the class logits.
 *
 * One definition, three executions. Under a graph context in the training phase it is trained (dropout on, the
 * tape records for autograd); under a plain CPU context it runs eagerly; under a tracing context it becomes the
 * StableHLO function a compiler takes. Nothing here knows which.
 *
 * Input `[batch, maxRows, dim + 1]`, output `[batch, classes]` — raw logits (`classification/v1`).
 */
public fun classificationHead(config: ClassificationHeadConfig, ctx: ExecutionContext, random: Random): Module<FP32, Float> {
    val classifier = sequential<FP32, Float>(ctx) {
        input(config.dim)
        dense(config.hidden, id = "hidden") {
            weights { randn(std = 0.05f, random = random) }
            bias { zeros() }
        }
        activation { it.relu() }
    }
    val logits = sequential<FP32, Float>(ctx) {
        input(config.hidden)
        dense(config.classes, id = "logits") {
            weights { randn(std = 0.05f, random = random) }
            bias { zeros() }
        }
    }
    return MLP(
        MaskedAttentionPool(config.dim, ctx, random),
        classifier,
        Dropout(p = config.dropout, name = "dropout", random = random),
        logits,
        name = "classification-head",
    )
}
