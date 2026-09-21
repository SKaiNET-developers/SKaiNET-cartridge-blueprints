package sk.ainet.cartridge.blueprint

import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.steps.HostGatherRewrite
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class HostGatherRewriteTest {
    private val module = """
        module {
          func.func @gemma_with_past(%arg0: tensor<1xi32>, %arg1: tensor<1x256xf32>) -> (tensor<1xi32>) {
            %v0 = stablehlo.constant dense_resource<model::embed> : tensor<262144x640xf32>
            %v237 = "stablehlo.gather"(%v0, %arg0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 640>, indices_are_sorted = false}> : (tensor<262144x640xf32>, tensor<1xi32>) -> tensor<1x640xf32>
            %v238 = stablehlo.multiply %v237, %v237 : tensor<1x640xf32>
            return %arg0 : tensor<1xi32>
          }
        }
    """.trimIndent() + "\n"

    @Test
    fun `the embedding gather becomes a function argument`() {
        val r = HostGatherRewrite.rewrite(module)
        assertEquals("gemma_with_past", r.function)
        assertEquals("tensor<1x640xf32>", r.embeddingType)
        assertEquals("%v0", r.table)
        assertContains(r.mlir, "func.func @gemma_with_past(%arg0: tensor<1xi32>, %emb: tensor<1x640xf32>, %arg1: tensor<1x256xf32>)")
        assertContains(r.mlir, "    %ez = stablehlo.constant dense<0.0> : tensor<1x640xf32>\n    %v237 = stablehlo.add %emb, %ez : tensor<1x640xf32>\n")
        assertFalse(r.mlir.contains("stablehlo.gather"))
        assertContains(r.mlir, "%v238 = stablehlo.multiply %v237, %v237", message = "downstream uses of the gather result are untouched")
    }

    @Test
    fun `a module without an embedding lookup on arg0 is refused`() {
        val e = assertFailsWith<MaterializationException> { HostGatherRewrite.rewrite(module.replace("%arg0)", "%arg1)")) }
        assertContains(e.message!!, "no `\"stablehlo.gather\"")
    }

    /**
     * Byte-for-byte parity with the rewrite the original cartridge build applied. Opt-in: set
     * BLUEPRINT_GOLDEN_MLIR=<dir> to directories holding `<name>.mlir` next to `<name>-hostgather.mlir`.
     */
    @Test
    fun `matches the reference rewrite on real exported modules`() {
        val roots = System.getenv("BLUEPRINT_GOLDEN_MLIR")?.split(File.pathSeparator)?.map(::File) ?: return
        var compared = 0
        for (root in roots) for (golden in root.walkTopDown().filter { it.name.endsWith("-hostgather.mlir") }) {
            val original = File(golden.parentFile, golden.name.replace("-hostgather.mlir", ".mlir"))
            if (!original.isFile) continue
            assertEquals(golden.readText(), HostGatherRewrite.rewrite(original.readText()).mlir, original.path)
            compared++
        }
        check(compared > 0) { "BLUEPRINT_GOLDEN_MLIR is set but no <name>.mlir / <name>-hostgather.mlir pairs were found" }
        println("host-gather golden parity: $compared module(s) byte-identical")
    }
}
