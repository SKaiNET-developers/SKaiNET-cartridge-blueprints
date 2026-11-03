package sk.ainet.cartridge.blueprint

import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.steps.CrossKvCacheRewrite
import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class CrossKvCacheRewriteTest {
    // A decoder prefill in small: 6 memory rows of 8, 2 heads of 4. Cross K/V per row from the memory, a query from
    // the token, attention against K (through a duplicate reshape, as the exporter emits them).
    private val prefill = """
        module {
          func.func @pre(%arg0: tensor<1x1x8xf32>, %arg1: tensor<1x6x8xf32>, %arg2: tensor<1x1x1x6xf32>) -> (tensor<1x2x1x6xf32>, tensor<1x2x6x4xf32>, tensor<1x2x6x4xf32>) {
            %v0 = stablehlo.constant dense<1.0> : tensor<8x8xf32>
            %v1 = stablehlo.constant dense<2.0> : tensor<8x8xf32>
            %v2 = stablehlo.constant dense<3.0> : tensor<8x8xf32>
            %v3 = stablehlo.transpose %v0, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
            %v4 = stablehlo.transpose %v1, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
            %v5 = stablehlo.dot_general %arg1, %v3, contracting_dims = [2] x [0] : (tensor<1x6x8xf32>, tensor<8x8xf32>) -> tensor<1x6x8xf32>
            %v6 = stablehlo.dot_general %arg1, %v4, contracting_dims = [2] x [0] : (tensor<1x6x8xf32>, tensor<8x8xf32>) -> tensor<1x6x8xf32>
            %v7 = stablehlo.reshape %v5 : (tensor<1x6x8xf32>) -> tensor<6x2x4xf32>
            %v8 = stablehlo.transpose %v7, dims = [1, 0, 2] : (tensor<6x2x4xf32>) -> tensor<2x6x4xf32>
            %v9 = stablehlo.reshape %v8 : (tensor<2x6x4xf32>) -> tensor<1x2x6x4xf32>
            %v10 = stablehlo.reshape %v8 : (tensor<2x6x4xf32>) -> tensor<1x2x6x4xf32>
            %v11 = stablehlo.reshape %v6 : (tensor<1x6x8xf32>) -> tensor<6x2x4xf32>
            %v12 = stablehlo.transpose %v11, dims = [1, 0, 2] : (tensor<6x2x4xf32>) -> tensor<2x6x4xf32>
            %v13 = stablehlo.reshape %v12 : (tensor<2x6x4xf32>) -> tensor<1x2x6x4xf32>
            %v14 = stablehlo.dot_general %arg0, %v2, contracting_dims = [2] x [0] : (tensor<1x1x8xf32>, tensor<8x8xf32>) -> tensor<1x1x8xf32>
            %v15 = stablehlo.reshape %v14 : (tensor<1x1x8xf32>) -> tensor<1x2x1x4xf32>
            %v16 = stablehlo.dot_general %v15, %v10, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x2x1x4xf32>, tensor<1x2x6x4xf32>) -> tensor<1x2x1x6xf32>
            return %v16, %v9, %v13 : tensor<1x2x1x6xf32>, tensor<1x2x6x4xf32>, tensor<1x2x6x4xf32>
          }
        }
    """.trimIndent() + "\n"

    private val adapter = """
        module {
          func.func @adp(%arg0: tensor<1x2xi32>, %arg1: tensor<1x2x8xf32>) -> (tensor<1x2x8xf32>) {
            %v0 = stablehlo.constant dense<0.5> : tensor<1x2x8xf32>
            %v1 = stablehlo.add %arg1, %v0 : tensor<1x2x8xf32>
            return %v1 : tensor<1x2x8xf32>
          }
        }
    """.trimIndent() + "\n"

    private fun run(pre: String = prefill): Triple<CrossKvCacheRewrite.Result, String, String> {
        val d = createTempDirectory("cross-kv").toFile()
        val (p, a) = File(d, "prefill.mlir").apply { writeText(pre) } to File(d, "adapter.mlir").apply { writeText(adapter) }
        val r = CrossKvCacheRewrite.rewrite(p, a, File(d, "prefill-kv.mlir"), File(d, "adapter-kv.mlir"))
        return Triple(r, File(d, "prefill-kv.mlir").readText(), File(d, "adapter-kv.mlir").readText())
    }

    @Test
    fun `prefill reads the cross K and V as arguments, its projections and their weights are gone`() {
        val (r, pre, _) = run()
        assertEquals(CrossKvCacheRewrite.Result(outputs = 2, memoryRows = 6, windowRows = 2, coneOps = 9), r)
        assertContains(pre, "func.func @pre(%arg0: tensor<1x1x8xf32>, %ckv0: tensor<1x2x6x4xf32>, %ckv1: tensor<1x2x6x4xf32>, %arg2: tensor<1x1x1x6xf32>) -> (tensor<1x2x1x6xf32>) {")
        assertContains(pre, "%v16 = stablehlo.dot_general %v15, %ckv0, batching_dims")
        assertContains(pre, "return %v16 : tensor<1x2x1x6xf32>")
        assertContains(pre, "%v2 = stablehlo.constant")
        for (gone in listOf("%v0 =", "%v1 =", "%v3 =", "%v5 =", "%v10 =", "%v13 =", "%arg1")) assertFalse(gone in pre, "$gone is gone")
    }

    @Test
    fun `the adapter computes the K and V of its window rows and returns them after its output`() {
        val (_, _, adp) = run()
        assertContains(adp, "func.func @adp(%arg0: tensor<1x2xi32>, %arg1: tensor<1x2x8xf32>) -> (tensor<1x2x8xf32>, tensor<1x2x2x4xf32>, tensor<1x2x2x4xf32>) {")
        assertContains(adp, "%kv_v0 = stablehlo.constant dense<1.0> : tensor<8x8xf32>")
        assertContains(adp, "%kv_v5 = stablehlo.dot_general %v1, %kv_v3, contracting_dims = [2] x [0] : (tensor<1x2x8xf32>, tensor<8x8xf32>) -> tensor<1x2x8xf32>")
        assertContains(adp, "%kv_v8 = stablehlo.transpose %kv_v7, dims = [1, 0, 2] : (tensor<2x2x4xf32>) -> tensor<2x2x4xf32>")
        assertContains(adp, "return %v1, %kv_v9, %kv_v13 : tensor<1x2x8xf32>, tensor<1x2x2x4xf32>, tensor<1x2x2x4xf32>")
        assertFalse("%kv_v2 =" in adp, "the query weight stays in prefill")
    }

    @Test
    fun `a memory-derived value prefill uses but does not return is refused`() {
        val e = assertFailsWith<MaterializationException> {
            run(prefill.replace("%v16 = stablehlo.dot_general %v15, %v10,", "%v16 = stablehlo.dot_general %v15, %v8x,").replace(
                "%v10 = stablehlo.reshape %v8 :", "%v8x = stablehlo.reshape %v12 :"))
        }
        assertContains(e.message!!, "is computed from the memory and used inside prefill, but not returned")
    }
}
