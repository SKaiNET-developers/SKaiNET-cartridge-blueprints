package sk.ainet.cartridge.blueprint

import sk.ainet.cartridge.blueprint.model.MaterializationException
import sk.ainet.cartridge.blueprint.steps.TorqBf16Retype
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class TorqBf16RetypeTest {
    // A folded block-quant matmul in small (i8 codes, bf16 scale), an RMS norm sum and a softmax max.
    private val module = """
        module {
          func.func @decode_step(%arg0: tensor<1x32xf32>) -> (tensor<1x4xf32>) {
            %v0 = stablehlo.constant dense<"0x0102030405060708"> : tensor<1x1x8xi8>
            %v1 = stablehlo.constant dense<"0x803F"> : tensor<1x1x1xbf16>
            %v2 = stablehlo.constant dense<0.0> : tensor<f32>
            %v3 = stablehlo.constant dense<[1.0, 0.5, 0.25, 0.125]> : tensor<4xf32>
            %v4 = stablehlo.reduce(%arg0 init: %v2) applies stablehlo.add across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
            %v5 = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %v6 = stablehlo.reduce(%arg0 init: %v5) applies stablehlo.maximum across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
            %v7 = linalg.generic {indexing_maps = [], iterator_types = []} ins(%v0, %v1 : tensor<1x1x8xi8>, tensor<1x1x1xbf16>) outs(%e : tensor<1x4xf32>) {
            ^bb0(%in: i8, %s: bf16, %out: f32):
              %c = arith.sitofp %in : i8 to f32
              linalg.yield %c : f32
            } -> tensor<1x4xf32>
            return %v7 : tensor<1x4xf32>
          }
        }
    """.trimIndent() + "\n"

    @Test
    fun everyF32BecomesBf16AndTheInfinityKeepsItsValue() {
        val (out, result) = TorqBf16Retype.retype(module)
        assertFalse("f32" in out, out)
        assertContains(out, "func.func @decode_step(%arg0: tensor<1x32xbf16>) -> (tensor<1x4xbf16>)")
        assertContains(out, "dense<0xFF80> : tensor<bf16>")
        assertContains(out, "dense<[1.0, 0.5, 0.25, 0.125]> : tensor<4xbf16>")
        assertContains(out, "%out: bf16")
        assertContains(out, "arith.sitofp %in : i8 to bf16")
        // Packed codes and bf16 scales are untouched.
        assertContains(out, "dense<\"0x0102030405060708\"> : tensor<1x1x8xi8>")
        assertContains(out, "dense<\"0x803F\"> : tensor<1x1x1xbf16>")
        assertEquals(1, result.infinities)
        assertEquals(module.lines().size - 1, result.lines)
        assertEquals(out.lines().size, module.lines().size)
    }

    @Test
    fun anOtherF32HexConstantFailsTheStep() {
        val blob = module.replace("dense<[1.0, 0.5, 0.25, 0.125]> : tensor<4xf32>", "dense<\"0x0000803F0000003F0000803E0000003E\"> : tensor<4xf32>")
        assertFailsWith<MaterializationException> { TorqBf16Retype.retype(blob) }
    }

    @Test
    fun aModuleWithoutF32IsRejected() {
        assertFailsWith<MaterializationException> { TorqBf16Retype.retype(TorqBf16Retype.retype(module).first) }
    }
}
