package sk.ainet.cartridge.classification.moonshine.head.recipe

import sk.ainet.cartridge.classification.moonshine.head.model.ClassificationHeadConfig
import sk.ainet.cartridge.classification.moonshine.head.model.classificationHead
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import kotlin.random.Random

/**
 * The third execution of the head: traced, not computed. The same [classificationHead] definition runs under a
 * tracing context with the trained weights; the tape becomes one StableHLO function
 * `rows[1, maxRows, dim + 1] -> logits[1, classes]` with the weights folded in as constants.
 */
internal object HeadExport {
    const val FUNCTION = "classify"

    fun stableHlo(config: ClassificationHeadConfig, weights: HeadWeights): String {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        // Dropout is the identity outside the training phase, so the traced graph has none.
        val model = classificationHead(config.copy(dropout = 0f), ctx, Random(0))
        weights.applyTo(model, ctx)
        val graph = trace(ctx) { model.forward(void(Shape(1, config.maxRows, config.rowWidth)), it) }
        val mlir = StableHloConverterFactory.createExtended().convert(graph, FUNCTION).content
        check("stablehlo.constant" in mlir) { "the trained weights did not fold into constants" }
        check("tensor<?" !in mlir && "x?x" !in mlir) { "the exported function has a dynamic shape" }
        val input = "tensor<1x${config.maxRows}x${config.rowWidth}xf32>"
        val output = "tensor<1x${config.classes}xf32>"
        check(input in mlir) { "the exported function does not take $input" }
        check(output in mlir) { "the exported function does not return $output" }
        return mlir
    }

    private fun trace(ctx: DefaultGraphExecutionContext, body: (ExecutionContext) -> Unit): ComputeGraph {
        val tape = ctx.record {
            val current = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(current)
            try { body(this as ExecutionContext) } finally { Execution.tapeStack.popTape() }
        }.first
        return (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true, embedConstants = true)
    }

    private fun void(shape: Shape): VoidOpsTensor<FP32, Float> = VoidOpsTensor(
        object : TensorData<FP32, Float> {
            override val shape: Shape = shape
            override fun get(vararg indices: Int): Float = 0.0f
            override fun set(vararg indices: Int, value: Float) {}
        },
        FP32::class,
    )
}
