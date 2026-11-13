package sk.ainet.cartridge.classification.moonshine.head

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
import kotlin.random.Random
import kotlin.test.Test
import sk.ainet.cartridge.classification.moonshine.head.recipe.HeadExport
import sk.ainet.cartridge.classification.moonshine.head.recipe.HeadWeights
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ClassificationHeadTest {
    private val config = ClassificationHeadConfig(classes = 4, dim = 16, maxRows = 12, hidden = 24, dropout = 0.1f)

    /** [n] examples: class c adds direction c to every valid row; padded rows hold large noise and a zero mask. */
    private fun batch(n: Int, rng: Random): Pair<FloatArray, IntArray> {
        val x = FloatArray(n * config.maxRows * config.rowWidth)
        val y = IntArray(n)
        for (i in 0 until n) {
            val c = rng.nextInt(config.classes); y[i] = c
            val valid = 3 + rng.nextInt(config.maxRows - 3)
            for (t in 0 until config.maxRows) {
                val at = (i * config.maxRows + t) * config.rowWidth
                if (t < valid) {
                    for (d in 0 until config.dim) x[at + d] = (rng.nextFloat() - 0.5f) * 0.6f + if (d % config.classes == c) 1f else 0f
                    x[at + config.dim] = 1f
                } else {
                    for (d in 0 until config.dim) x[at + d] = (rng.nextFloat() - 0.5f) * 50f   // must not matter
                }
            }
        }
        return x to y
    }

    @Test
    fun `the head trains on masked rows and ignores the padding`() {
        val cpu = DirectCpuExecutionContext()
        val trainCtx = DefaultGraphExecutionContext(baseOps = cpu.ops, phase = Phase.TRAIN, createTapeFactory = { _ -> DefaultGradientTape() })
        val rng = Random(7)
        val model = classificationHead(config, trainCtx, rng)
        val runner = training<FP32, Float> {
            model { model }
            loss { CrossEntropyLoss() }
            optimizer { adam(lr = 0.01).apply { model.trainableParameters().forEach { addParameter(it) } } }
        }
        var first = 0f; var last = 0f
        repeat(120) { step ->
            val (x, y) = batch(32, rng)
            val loss = runner.step(
                trainCtx,
                trainCtx.fromFloatArray(Shape(32, config.maxRows, config.rowWidth), FP32::class, x),
                trainCtx.fromIntArray<sk.ainet.lang.types.Int32, Int>(Shape(32), sk.ainet.lang.types.Int32::class, y),
            ).data.get()
            if (step == 0) first = loss
            last = loss
        }
        assertTrue(last < first * 0.2f, "loss $first -> $last")

        val (x, y) = batch(200, Random(99))
        val logits = model.forward(cpu.fromFloatArray(Shape(200, config.maxRows, config.rowWidth), FP32::class, x), cpu)
        var correct = 0
        for (i in 0 until 200) {
            var best = 0
            for (c in 1 until config.classes) if (logits.data.get(i, c) > logits.data.get(i, best)) best = c
            if (best == y[i]) correct++
        }
        assertTrue(correct >= 190, "accuracy $correct/200")

        // The trained head survives its file, and the same definition traces to one static StableHLO function.
        val file = java.io.File.createTempFile("head", ".safetensors").apply { deleteOnExit() }
        HeadWeights.of(model).write(file, config)
        val restored = classificationHead(HeadWeights.config(file), cpu, Random(1))
        HeadWeights.read(file).applyTo(restored, cpu)
        val again = restored.forward(cpu.fromFloatArray(Shape(200, config.maxRows, config.rowWidth), FP32::class, x), cpu)
        for (i in 0 until 200) for (c in 0 until config.classes) {
            assertEquals(logits.data.get(i, c), again.data.get(i, c), 1e-5f)
        }
        val mlir = HeadExport.stableHlo(HeadWeights.config(file), HeadWeights.read(file))
        assertTrue("func.func @${HeadExport.FUNCTION}" in mlir || "@${HeadExport.FUNCTION}(" in mlir, mlir.take(400))
    }
}
