package sk.ainet.cartridge.blueprint.steps

import sk.ainet.cartridge.blueprint.model.MaterializationException

/**
 * A `convert` step on exported StableHLO: move the token-embedding lookup out of the graph.
 *
 * An exported decoder starts with `%vN = "stablehlo.gather"(%table, %arg0) … -> tensor<SxD>` — a gather of the
 * (vocab × dim) embedding table by the token ids in `%arg0`. On small GPUs that table is the single largest buffer
 * the graph touches, for a lookup the host can do by reading rows straight from the parameter archive. The rewrite
 * replaces the gather with a new function argument `%emb : tensor<SxD>` that the runtime fills with those rows.
 * Numerics are unchanged: the graph receives exactly the rows the gather would have produced.
 *
 * Text-level on purpose — it must be reviewable next to the exported module, and it changes three lines.
 */
object HostGatherRewrite {
    private val GATHER = Regex("""(?m)^( *)(%v\d+) = "stablehlo\.gather"\((%v\d+), %arg0\)[^\n]*-> (tensor<[^>]+>)\n""")
    private val SIGNATURE = Regex("""func\.func @(\w+)\(%arg0: (tensor<[^>]+>),""")

    data class Result(val mlir: String, val function: String, val embeddingType: String, val table: String)

    fun rewrite(mlir: String): Result {
        val gather = GATHER.find(mlir) ?: throw MaterializationException(
            "host-gather: no `\"stablehlo.gather\"(<table>, %arg0)` found — this module does not start with a token-embedding " +
                "lookup on its first argument, or the exporter's output format changed.",
        )
        val (indent, result, table, shape) = gather.destructured
        val withoutGather = mlir.replaceRange(
            gather.range,
            "$indent%ez = stablehlo.constant dense<0.0> : $shape\n$indent$result = stablehlo.add %emb, %ez : $shape\n",
        )
        val signature = SIGNATURE.find(withoutGather) ?: throw MaterializationException(
            "host-gather: no `func.func @<name>(%arg0: tensor<…>,` signature found to extend.",
        )
        val (function, idsType) = signature.destructured
        val rewritten = withoutGather.replaceRange(signature.range, "func.func @$function(%arg0: $idsType, %emb: $shape,")
        return Result(rewritten, function, shape, table)
    }
}
