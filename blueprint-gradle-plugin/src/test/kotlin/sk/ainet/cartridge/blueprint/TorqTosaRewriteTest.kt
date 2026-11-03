package sk.ainet.cartridge.blueprint

import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.steps.TorqTosaRewrite
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class TorqTosaRewriteTest {
    // a layer norm's tail and a softmax normalize, as the exporter writes them
    private val divides = """
        func.func @f(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x1xf32>, %arg2: tensor<1x4x1xf32>) -> tensor<1x4x8xf32> {
          %v1 = stablehlo.sqrt %arg1 : tensor<1x4x1xf32>
          %v2 = stablehlo.broadcast_in_dim %v1, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
          %v3 = stablehlo.divide %arg0, %v2 : tensor<1x4x8xf32>
          %v4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
          %v5 = stablehlo.divide %v3, %v4 : tensor<1x4x8xf32>
          return %v5 : tensor<1x4x8xf32>
        }
    """.trimIndent()

    @Test
    fun `a divide by a broadcast becomes a multiply by the broadcast reciprocal, rsqrt for a sqrt`() {
        val (out, n) = TorqTosaRewrite.prepareDivides(divides)
        assertEquals(2, n)
        assertContains(out, "%tq_r1 = stablehlo.rsqrt %arg1 : tensor<1x4x1xf32>")
        assertContains(out, "%tq_b1 = stablehlo.broadcast_in_dim %tq_r1, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>")
        assertContains(out, "%v3 = stablehlo.multiply %arg0, %tq_b1 : tensor<1x4x8xf32>")
        assertContains(out, "%tq_r2 = stablehlo.divide %tq_one2, %arg2 : tensor<1x4x1xf32>")
        assertContains(out, "%v5 = stablehlo.multiply %v3, %tq_b2 : tensor<1x4x8xf32>")
    }

    // what --stablehlo-legalize-to-tosa leaves: broadcasts, an rsqrt, a weight matmul and attention's two batched ones
    private val legalized = """
        func.func @f(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x1xf32>, %arg2: tensor<8x6xf32>, %q: tensor<1x2x4x3xf32>, %k: tensor<1x2x4x3xf32>, %p: tensor<1x2x4x4xf32>) -> tensor<1x4x8xf32> {
          %0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
          %1 = stablehlo.rsqrt %arg1 : tensor<1x4x1xf32>
          %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
          %3 = tosa.mul %arg0, %2, %0 : (tensor<1x4x8xf32>, tensor<1x4x8xf32>, tensor<1xi8>) -> tensor<1x4x8xf32>
          %4 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
          %5 = tosa.reduce_sum %4 {axis = 2 : i32} : (tensor<1x4x8xf32>) -> tensor<1x4x1xf32>
          %6 = stablehlo.dot_general %3, %arg2, contracting_dims = [2] x [0] : (tensor<1x4x8xf32>, tensor<8x6xf32>) -> tensor<1x4x6xf32>
          %7 = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x2x4x3xf32>, tensor<1x2x4x3xf32>) -> tensor<1x2x4x4xf32>
          %8 = stablehlo.dot_general %p, %q, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x2x4x4xf32>, tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>
          return %3 : tensor<1x4x8xf32>
        }
    """.trimIndent()

    @Test
    fun `broadcasts feeding elementwise ops stay rank-equal, other broadcasts tile`() {
        val r = TorqTosaRewrite.finish(legalized)
        assertEquals(2, r.broadcasts); assertEquals(1, r.tiled); assertEquals(1, r.roots)
        assertContains(r.mlir, "%1 = tosa.rsqrt %arg1 : (tensor<1x4x1xf32>) -> tensor<1x4x1xf32>")
        // %2 only feeds the mul: no tile, the mul reads the [1,4,1] value and broadcasts it itself
        assertContains(r.mlir, "%tq_bc2 = tosa.identity %1 : (tensor<1x4x1xf32>) -> tensor<1x4x1xf32>")
        assertContains(r.mlir, "%3 = tosa.mul %arg0, %tq_bc2, %0 : (tensor<1x4x8xf32>, tensor<1x4x1xf32>, tensor<1xi8>) -> tensor<1x4x8xf32>")
        // %4 feeds a reduce: tiled to the full shape
        assertContains(r.mlir, "values = dense<[1, 1, 8]> : tensor<3xindex>")
        assertContains(r.mlir, "%4 = tosa.tile %arg1, ")
        assertFalse("stablehlo." in r.mlir, "nothing of StableHLO is left")
    }

    @Test
    fun `dot_generals become tosa matmuls, a weight reshaped to batch 1, attention per head`() {
        val r = TorqTosaRewrite.finish(legalized)
        assertEquals(1, r.weightMatmuls); assertEquals(2, r.batchedMatmuls)
        assertContains(r.mlir, "%tq_zp = \"tosa.const\"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>")
        assertContains(r.mlir, "-> tensor<1x8x6xf32>")
        assertContains(r.mlir, "%6 = tosa.matmul %3, %tq_rs")
        assertContains(r.mlir, ": (tensor<1x4x8xf32>, tensor<1x8x6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x6xf32>")
        // q · kᵀ: k transposed in its last two dimensions, then [2,4,3] x [2,3,4]
        assertContains(r.mlir, "tosa.transpose %k {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x4x3xf32>) -> tensor<1x2x3x4xf32>")
        assertContains(r.mlir, ": (tensor<2x4x3xf32>, tensor<2x3x4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x4x4xf32>")
        assertContains(r.mlir, "%7 = tosa.reshape %tq_bm")
        // p · v: [2,4,4] x [2,4,3]
        assertContains(r.mlir, ": (tensor<2x4x4xf32>, tensor<2x4x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x4x3xf32>")
    }

    @Test
    fun `a StableHLO op the step does not translate fails it`() {
        val e = assertFailsWith<MaterializationException> {
            TorqTosaRewrite.finish(legalized.replace("%5 = tosa.reduce_sum %4 {axis = 2 : i32} : (tensor<1x4x8xf32>) -> tensor<1x4x1xf32>",
                "%5 = stablehlo.cosine %4 : tensor<1x4x8xf32>"))
        }
        assertContains(e.message!!, "the legalization left `stablehlo.cosine`")
    }
}
