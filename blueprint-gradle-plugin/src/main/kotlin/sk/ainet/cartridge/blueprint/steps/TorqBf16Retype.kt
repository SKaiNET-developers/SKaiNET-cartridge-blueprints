package sk.ainet.cartridge.blueprint.steps

import sk.ainet.cartridge.blueprint.model.MaterializationException
import java.io.BufferedReader
import java.io.StringReader
import java.io.StringWriter
import java.io.Writer

/**
 * A `convert` step on exported StableHLO whose matmuls are already folded block-quant kernels (`linalg.generic` over
 * i8/i4 codes with bf16 scales, as SKaiNET's block-quant matmul fusion emits them): every f32 in the module becomes
 * bf16, function arguments and results included, and nothing else changes. Torq runs f32 activations through its
 * host fallback, which returns NaN; [TorqBf16Rewrite] is the route for graphs with dense f32 weight matmuls, and it
 * requires them.
 *
 * Constants stay valid as they are: decimal literals parse as bf16, and the only f32 hex literals allowed are the
 * scalar infinities (`0xFF800000`, `0x7F800000`, the max-reduction initial values), rewritten to their bf16 bits. Any
 * other f32 hex constant would change meaning under a retype, so it fails the step.
 *
 * Streams line by line: a decode step with inline packed weights is several hundred MB of text.
 * Compile the output with the Torq backend's `stablehlo-torq` input type ([IreeCompileTask] `torq-stablehlo`).
 */
object TorqBf16Retype {
    data class Result(val lines: Int, val retypedLines: Int, val infinities: Int)

    private val f32Hex = Regex("""dense<"?0x([0-9A-Fa-f]+)"?>\s*:\s*tensor<[^>]*f32>""")
    // The f32 type token: in `tensor<…xf32>`, `tensor<f32>`, `: f32`, `to f32`, `-> f32`; not inside a name.
    private val f32Type = Regex("""(?<![A-Za-z0-9_%@#$.])f32\b|(?<=\dx|\?x)f32\b""")
    private val infinities = mapOf("FF800000" to "0xFF80", "7F800000" to "0x7F80")

    fun retype(input: BufferedReader, output: Writer): Result {
        var lines = 0
        var retyped = 0
        var inf = 0
        while (true) {
            var line = input.readLine() ?: break
            lines++
            if ("f32" in line) {
                f32Hex.find(line)?.let { m ->
                    val bits = m.groupValues[1].uppercase()
                    val bf16 = infinities[bits] ?: throw MaterializationException(
                        "torq-bf16-retype: line $lines has an f32 hex constant (0x${bits.take(16)}…) that a retype " +
                            "would reinterpret; export it as decimal or use the torq-bf16 rewrite.",
                    )
                    line = line.replaceFirst(m.value, m.value.replace("0x${m.groupValues[1]}", bf16))
                    inf++
                }
                line = f32Type.replace(line, "bf16")
                retyped++
            }
            output.write(line)
            output.write("\n")
        }
        if (retyped == 0) throw MaterializationException("torq-bf16-retype: the module has no f32 types; nothing to retype.")
        return Result(lines, retyped, inf)
    }

    fun retype(mlir: String): Pair<String, Result> {
        val out = StringWriter()
        val result = retype(BufferedReader(StringReader(mlir)), out)
        return out.toString().removeSuffix("\n").let { if (mlir.endsWith("\n")) "$it\n" else it } to result
    }
}
