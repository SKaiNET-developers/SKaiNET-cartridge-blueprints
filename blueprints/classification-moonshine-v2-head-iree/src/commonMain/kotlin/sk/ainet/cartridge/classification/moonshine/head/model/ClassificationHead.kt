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
    val dropout: Float = 0.3f,
    /** Attention poolings side by side; each owns `hidden / heads` of the hidden features. */
    val heads: Int = 8,
) {
    /** One input row: [dim] features and, last, the validity mask (1 for an encoder row, 0 for padding). */
    val rowWidth: Int get() = dim + 1

    init {
        require(classes >= 2) { "a classifier needs at least two classes, got $classes" }
        require(dim > 0 && maxRows > 0 && hidden > 0 && heads > 0) { "dim, maxRows, hidden and heads must be positive" }
        require(hidden % heads == 0) { "hidden ($hidden) must be a multiple of heads ($heads): each head owns an equal share" }
        require(dropout in 0f..0.9f) { "dropout must be in [0, 0.9], got $dropout" }
    }
}

/**
 * Attention pooling over the valid rows, [heads] times side by side. Each head has its own [unit] features of a
 * row and its own learned score per row; it takes a softmax of its scores over the rows that are there and the
 * weighted sum of its features. The heads learn different views of which rows matter. The output is rectified:
 * it is the head's hidden layer.
 *
 * Input `[batch, rows, dim + 1]` — the last column is the mask. Output `[batch, heads · unit]`. A padded row gets
 * `score − 1e9` before the softmax, so its weight is exactly zero in float32 whatever the row holds; there is no
 * `where`, only arithmetic that every backend and the exporter share.
 *
 * Features and scores read the whole input row, mask column included: on a real row that column is 1, so its
 * weight is one more bias, and a padded row does not matter.
 *
 * Everything large is a matrix product or a product of two equal shapes: the weights reach their head's features
 * through a constant 0/1 matrix, not a broadcast.
 */
public class MaskedAttentionPool(
    private val dim: Int,
    private val heads: Int,
    private val unit: Int,
    ctx: ExecutionContext,
    random: Random,
    override val name: String = "pool",
) : Module<FP32, Float>(), ModuleParameters<FP32, Float> {

    private val width = dim + 1
    private val features = heads * unit

    private val rows = Linear<FP32, Float>(
        inFeatures = width,
        outFeatures = features,
        name = "$name.rows",
        initWeights = ctx.fromFloatArray(Shape(features, width), FP32::class, FloatArray(features * width) { (random.nextFloat() - 0.5f) * 0.1f }),
        initBias = ctx.fromFloatArray(Shape(features), FP32::class, FloatArray(features)),
    )

    private val score = Linear<FP32, Float>(
        inFeatures = width,
        outFeatures = heads,
        name = "$name.score",
        initWeights = ctx.fromFloatArray(Shape(heads, width), FP32::class, FloatArray(heads * width) { (random.nextFloat() - 0.5f) * 0.02f }),
        initBias = ctx.fromFloatArray(Shape(heads), FP32::class, FloatArray(heads)),
    )

    override val modules: List<Module<FP32, Float>> get() = listOf(rows, score)
    override val params: List<ModuleParameter<FP32, Float>> get() = emptyList()

    override fun onForward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val (batch, count) = input.shape.dimensions
        val flat = input.reshape(Shape(batch * count, width))
        // −1e9 on every head's score of a padded row: the mask column, spread across the heads by a row of ones.
        val padded = (flat.narrow(dim = 1, start = dim, length = 1) - 1f) * MASKED       // [b·t, 1]
        val ones = ctx.fromFloatArray<FP32, Float>(Shape(1, heads), FP32::class, FloatArray(heads) { 1f })
        val scores = score.forward(flat, ctx) + ctx.ops.matmul(padded, ones)             // [b·t, heads]
        // The softmax runs over the rows, so the rows go last for it and come back after.
        val weights = ctx.ops.permute(
            ctx.ops.permute(scores.reshape(Shape(batch, count, heads)), SWAP)            // [b, heads, t]
                .reshape(Shape(batch * heads, count)).softmax(dim = 1)
                .reshape(Shape(batch, heads, count)),
            SWAP,
        ).reshape(Shape(batch * count, heads))                                           // [b·t, heads]
        // Each head's weight on each of its own features: head h owns features h·unit until (h + 1)·unit.
        val owner = ctx.fromFloatArray<FP32, Float>(
            Shape(heads, features), FP32::class,
            FloatArray(heads * features) { if ((it % features) / unit == it / features) 1f else 0f },
        )
        val weighted = rows.forward(flat, ctx) * ctx.ops.matmul(weights, owner)          // [b·t, heads·unit]
        return weighted.reshape(Shape(batch, count, features)).sum(dim = 1).relu()       // [b, heads·unit]
    }

    private companion object {
        const val MASKED = 1e9f

        /** Rows and heads change places; the batch stays first. */
        val SWAP = intArrayOf(0, 2, 1)
    }
}

/**
 * The head: masked attention poolings as the hidden layer, then the class logits.
 *
 * One definition, three executions. Under a graph context in the training phase it is trained (dropout on, the
 * tape records for autograd); under a plain CPU context it runs eagerly; under a tracing context it becomes the
 * StableHLO function a compiler takes. Nothing here knows which.
 *
 * Input `[batch, maxRows, dim + 1]`, output `[batch, classes]` — raw logits (`classification/v1`).
 */
public fun classificationHead(config: ClassificationHeadConfig, ctx: ExecutionContext, random: Random): Module<FP32, Float> {
    val logits = sequential<FP32, Float>(ctx) {
        input(config.hidden)
        dense(config.classes, id = "logits") {
            weights { randn(std = 0.05f, random = random) }
            bias { zeros() }
        }
    }
    return MLP(
        MaskedAttentionPool(config.dim, config.heads, config.hidden / config.heads, ctx, random),
        Dropout(p = config.dropout, name = "dropout", random = random),
        logits,
        name = "classification-head",
    )
}
