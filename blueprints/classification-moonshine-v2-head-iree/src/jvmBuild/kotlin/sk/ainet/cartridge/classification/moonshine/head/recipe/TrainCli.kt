package sk.ainet.cartridge.classification.moonshine.head.recipe

import kotlinx.serialization.Serializable
import sk.ainet.cartridge.classification.moonshine.head.Decision
import sk.ainet.cartridge.classification.moonshine.head.LabelSet
import sk.ainet.cartridge.classification.moonshine.head.model.ClassificationHeadConfig
import sk.ainet.cartridge.classification.moonshine.head.model.classificationHead
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.dsl.training
import sk.ainet.lang.nn.loss.CrossEntropyLoss
import sk.ainet.lang.nn.optim.adam
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.math.sqrt
import kotlin.random.Random

@Serializable
internal data class TrainReport(
    val seed: Long,
    val classes: Int,
    val train: Int,
    val dev: Int,
    val epochs: Int,
    val bestEpoch: Int,
    val devTop1: Double,
    val history: List<Epoch>,
) {
    @Serializable
    data class Epoch(val epoch: Int, val loss: Double, val devTop1: Double)
}

/**
 * Step `train` (ADR-018): the first execution of the head. The definition in `model/ClassificationHead.kt` is
 * fitted with the SKaiNET training DSL on the `train` split; the epoch with the best `dev` top-1 is kept.
 *
 *   --features <dir> --label-set <label-set.json> --out <dir> --seed <n>
 *   [--hidden 256] [--dropout 0.1] [--learning-rate 0.001] [--batch 64] [--max-epochs 40] [--patience 6] [--max-rows 256]
 *
 * Everything random comes from `--seed`: the initial weights, dropout, and the order of the examples.
 * Writes `head.safetensors`, `label-set.json` (the pack's label set) and `train-report.json`.
 */
public fun main(args: Array<String>) {
    val a = Args(args)
    val seed = a.long("seed", -1).also { require(it >= 0) { "missing --seed: a train step states its seed (ADR-018)" } }
    val labels = LabelSetInput.read(a.file("label-set"))
    val store = FeatureStore(a.file("features"))
    val out = a.file("out").apply { mkdirs() }
    val header = store.header() ?: error("no features in ${a.file("features")}")
    val batch = a.int("batch", 64)
    val maxEpochs = a.int("max-epochs", 40)
    val patience = a.int("patience", 6)

    val train = store.examples("train", labels).filter { it.target >= 0 }
    val dev = store.examples("dev", labels).filter { it.target >= 0 }
    require(train.isNotEmpty()) { "the dataset has no usable `train` recordings for this label set" }
    require(dev.isNotEmpty()) { "the dataset has no usable `dev` recordings for this label set: the epoch is picked on dev" }
    val byClass = train.groupBy { it.target }
    val missing = labels.labels.indices.filter { it !in byClass }.map { labels.labels[it] }
    require(missing.isEmpty()) { "no `train` recordings for: $missing" }

    val config = ClassificationHeadConfig(
        classes = labels.labels.size, dim = header.dim, maxRows = a.int("max-rows", header.maxRows),
        hidden = a.int("hidden", 256), dropout = a.float("dropout", 0.1f),
    )
    val random = Random(seed)
    val cpu = DirectCpuExecutionContext()
    val ctx = DefaultGraphExecutionContext(baseOps = cpu.ops, phase = Phase.TRAIN, createTapeFactory = { _ -> DefaultGradientTape() })
    val model = classificationHead(config, ctx, random)
    val runner = training<FP32, Float> {
        model { model }
        loss { CrossEntropyLoss() }
        optimizer { adam(lr = a.float("learning-rate", 0.001f).toDouble()).apply { model.trainableParameters().forEach { addParameter(it) } } }
    }
    val eager = EagerHead(config, model, cpu)

    // Class-balanced sampling: a class is drawn in proportion to the square root of its size, so a rare class is
    // heard often enough without the head forgetting how common the common ones are.
    val classes = byClass.keys.sorted()
    val weight = classes.map { sqrt(byClass.getValue(it).size.toDouble()) }
    val total = weight.sum()
    fun draw(): Example {
        var r = random.nextDouble() * total
        for ((i, c) in classes.withIndex()) { r -= weight[i]; if (r <= 0) return byClass.getValue(c).random(random) }
        return byClass.getValue(classes.last()).random(random)
    }

    println("train: ${train.size} train, ${dev.size} dev, ${config.classes} classes, seed $seed")
    val history = ArrayList<TrainReport.Epoch>()
    var best = -1.0; var bestEpoch = 0; var bestWeights = HeadWeights.of(model)
    val steps = maxOf(1, train.size / batch)
    for (epoch in 1..maxEpochs) {
        var loss = 0.0
        repeat(steps) {
            val examples = List(batch) { draw() }
            val rows = minOf(config.maxRows, examples.maxOf { it.count })
            loss += runner.step(
                ctx,
                ctx.fromFloatArray(Shape(batch, rows, config.rowWidth), FP32::class, pack(examples, config.dim, rows)),
                ctx.fromIntArray<Int32, Int>(Shape(batch), Int32::class, IntArray(batch) { examples[it].target }),
            ).data.get()
            ctx.clearSession()   // the trace session keeps every tensor of a step alive until it is cleared
        }
        val top1 = top1(eager.logits(dev), dev)
        history += TrainReport.Epoch(epoch, loss / steps, top1)
        println("train: epoch $epoch  loss ${"%.4f".format(loss / steps)}  dev top-1 ${"%.4f".format(top1)}")
        if (top1 > best) { best = top1; bestEpoch = epoch; bestWeights = HeadWeights.of(model) }
        if (epoch - bestEpoch >= patience) break
    }

    HeadWeights(bestWeights.tensors, mapOf("seed" to seed.toString(), "epoch" to bestEpoch.toString())).write(out.resolve("head.safetensors"), config)
    out.resolve("label-set.json").writeText(json.encodeToString(LabelSet.serializer(), labels.labelSet) + "\n")
    out.resolve("train-report.json").writeText(
        json.encodeToString(TrainReport.serializer(), TrainReport(seed, config.classes, train.size, dev.size, history.size, bestEpoch, best, history)) + "\n",
    )
    println("train: kept epoch $bestEpoch, dev top-1 ${"%.4f".format(best)}")
}

internal fun top1(logits: List<FloatArray>, examples: List<Example>): Double =
    logits.indices.count { Decision.argmax(logits[it]) == examples[it].target }.toDouble() / examples.size
