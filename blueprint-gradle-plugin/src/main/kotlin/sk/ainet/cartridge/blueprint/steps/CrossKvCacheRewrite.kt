package sk.ainet.cartridge.blueprint.steps

import sk.ainet.cartridge.blueprint.model.MaterializationException
import java.io.File
import java.io.Writer

/**
 * A `convert` step on the exported streaming decoder: the cross-attention K/V of each memory row is computed once, by
 * one matmul over the window's rows, instead of by every prefill over the whole padded memory.
 *
 * The prefill graph projects its memory argument (`[1, M, D]`, M = 256 padded rows) to every layer's cross K and V
 * (`[1, H, M, d]`) with per-row matmuls, reshapes and transposes: the "memory cone". Those ops see one row at a time,
 * so a row's K/V never changes once the row is final, yet the streaming loop runs prefill on every restart and hop,
 * and the cone is ~93% of its work (Moonshine v2 tiny: 629 of 674 MFLOP).
 *
 * - prefill: the memory argument is replaced by the cone's outputs, one argument per distinct value, in the order the
 *   graph returned them (layer 0 K, layer 0 V, …); the cone, its weights and those outputs are dropped.
 * - encoder module: gets a second function, `<prefix>cross_kv` (the adapter's name with `adapter` replaced), from the
 *   adapter's output (`[1, C, D]`, C = window rows) to `[1, C, n·H·d]`: all n K/V projections as one matmul, their
 *   weights side by side, output j in columns `[j·H·d, (j+1)·H·d)`, head h of it at `h·d`. Each cone output must be
 *   `reshape(transpose(reshape(dot_general(memory, W)), [1, 0, 2]))` with a constant W (or its transpose).
 *
 * One matmul instead of n, and no head transposes: on the Torq NPU those run as host fallback, and the adapter's
 * positional gather is host code too (~20 ms per window), so the adapter stays where it was and only the projection
 * moves next to the encoder. The caller keeps a `[H, M, d]` cache per layer and side, copies the finalized rows of
 * each window in (splitting the heads), and passes the cache to prefill and the decoder steps. The decoder-step
 * graph is unchanged. Streams prefill line by line: an exported prefill is ~300 MB of text, single constants several MB.
 */
object CrossKvCacheRewrite {
    data class Result(val outputs: Int, val memoryRows: Int, val windowRows: Int, val coneOps: Int, val function: String)

    private const val LONG = 4096  // longer lines are constant literals: no uses in them
    private val NAME = Regex("""%[\w]+""")
    private val DEF = Regex("""^( *)(%[\w]+) = (.*)$""")
    private val SIG = Regex("""^( *func\.func @[\w]+)\((.*)\) -> \((.*)\) \{$""")
    private val ARG = Regex("""(%arg\d+): (tensor<[^>]+>)""")
    private val RETURN = Regex("""^( *)return (.+) : (.+)$""")
    private val MEMORY = Regex("""tensor<1x(\d+)x(\d+)xf32>""")
    private val RESHAPE = Regex("""^stablehlo\.reshape (%[\w]+) : \(tensor<[^>]+>\) -> tensor<([0-9x]+)xf32>$""")
    private val TRANSPOSE = Regex("""^stablehlo\.transpose (%[\w]+), dims = \[([\d, ]+)\] : \(tensor<[^>]+>\) -> tensor<[^>]+>$""")
    private val DOT = Regex("""^stablehlo\.dot_general (%[\w]+), (%[\w]+), contracting_dims = \[2\] x \[0\] : \(tensor<1x\d+x\d+xf32>, tensor<(\d+)x(\d+)xf32>\) -> tensor<1x\d+x\d+xf32>$""")
    private val CONSTANT = Regex("""^stablehlo\.constant dense<(.*)> : tensor<(\d+)x(\d+)xf32>$""")

    fun rewrite(prefill: File, adapter: File, encoder: File, prefillOut: File, encoderOut: File): Result {
        // pass 1: the prefill's ops (constant literals only by name)
        val defs = LinkedHashMap<String, String>()  // name -> right-hand side (long constants: "constant")
        var sig: MatchResult? = null
        var ret: MatchResult? = null
        prefill.bufferedReader().useLines { lines ->
            for (l in lines) {
                if (l.length > LONG) { DEF.find(l.substring(0, 256))?.let { defs[it.groupValues[2]] = "stablehlo.constant" }; continue }
                if (sig == null) SIG.matchEntire(l)?.let { sig = it }
                RETURN.matchEntire(l)?.let { ret = it }
                DEF.matchEntire(l)?.let { defs[it.groupValues[2]] = it.groupValues[3] }
            }
        }
        val s = sig ?: throw MaterializationException("cross-kv: no function signature in ${prefill.name}")
        val r = ret ?: throw MaterializationException("cross-kv: no return in ${prefill.name}")
        val args = ARG.findAll(s.groupValues[2]).map { it.groupValues[1] to it.groupValues[2] }.toList()
        val (mem, memType) = args.firstOrNull { (_, t) -> MEMORY.matchEntire(t)?.groupValues?.get(1)?.toInt()?.let { it > 1 } == true }
            ?: throw MaterializationException("cross-kv: ${prefill.name} has no [1, M, D] memory argument")
        val (m, dim) = MEMORY.matchEntire(memType)!!.destructured
        val rows = m.toInt()

        // the cone: ops on the memory and constants only; static: ops on constants only
        val static = HashSet<String>()
        val cone = LinkedHashSet<String>()
        for ((name, rhs) in defs) {
            val uses = operands(rhs)
            if (uses.all { it in static }) static += name
            else if (uses.any { it == mem || it in cone } && uses.all { it == mem || it in cone || it in static }) cone += name
        }
        // its outputs: cone values used outside it, one per distinct expression
        val canonical = HashMap<String, String>()  // cone value -> the first value with the same right-hand side
        val byRhs = HashMap<String, String>()
        for (v in cone) canonical[v] = byRhs.getOrPut(defs.getValue(v)) { v }
        val retValues = r.groupValues[2].split(", ")
        val retTypes = r.groupValues[3].split(", ")
        val outside = defs.filterKeys { it !in cone }.values.flatMap(::operands).filter { it in cone }.map(canonical::getValue).toSet()
        val boundary = retValues.filter { it in cone }.map(canonical::getValue).distinct()
        (outside - boundary.toSet()).firstOrNull()?.let {
            throw MaterializationException("cross-kv: $it is computed from the memory and used inside prefill, but not returned")
        }
        if (boundary.isEmpty()) throw MaterializationException("cross-kv: ${prefill.name} returns nothing computed from $mem")
        val newArg = boundary.withIndex().associate { (i, v) -> v to "%ckv$i" }
        val argOf = cone.filter { canonical.getValue(it) in newArg }.associateWith { newArg.getValue(canonical.getValue(it)) }
        val boundaryType = boundary.associateWith { v -> retTypes[retValues.indexOfFirst { canonical[it] == v }] }

        // the static values the cone needs (weights and their transposes); prefill drops those only the cone uses
        val needed = LinkedHashSet<String>()
        fun need(v: String) { if (v in static && needed.add(v)) operands(defs.getValue(v)).forEach(::need) }
        cone.forEach { c -> operands(defs.getValue(c)).forEach(::need) }

        val projections = boundary.map { projection(it, mem, defs) }
        val (window, function) = writeEncoder(encoder, encoderOut, adapter, prefill, dim.toInt(), projections)
        writePrefill(prefill, prefillOut, s, r, args, mem, cone, needed, defs, argOf, newArg, boundaryType)
        return Result(boundary.size, rows, window, cone.size, function)
    }

    /** One cone output's weight: the constant's name, and whether the matmul reads it transposed (`[out, in]`). */
    private data class Projection(val weight: String, val transposed: Boolean, val inputs: Int, val outputs: Int)

    private fun projection(v: String, mem: String, defs: Map<String, String>): Projection {
        fun fail(): Nothing = throw MaterializationException(
            "cross-kv: $v is not reshape(transpose(reshape(dot_general(memory, weight)), [1, 0, 2])); cannot merge it",
        )
        val outer = RESHAPE.matchEntire(defs.getValue(v)) ?: fail()
        val heads = TRANSPOSE.matchEntire(defs[outer.groupValues[1]] ?: fail())?.takeIf { it.groupValues[2] == "1, 0, 2" } ?: fail()
        val inner = RESHAPE.matchEntire(defs[heads.groupValues[1]] ?: fail()) ?: fail()
        val dot = DOT.matchEntire(defs[inner.groupValues[1]] ?: fail())?.takeIf { it.groupValues[1] == mem } ?: fail()
        val (w, rows, cols) = dot.destructured.let { (_, w, r, c) -> Triple(w, r.toInt(), c.toInt()) }
        val t = TRANSPOSE.matchEntire(defs[w] ?: fail())
        return when {
            t != null && t.groupValues[2] == "1, 0" && defs[t.groupValues[1]]?.startsWith("stablehlo.constant") == true ->
                Projection(t.groupValues[1], transposed = true, inputs = rows, outputs = cols)
            defs[w]?.startsWith("stablehlo.constant") == true -> Projection(w, transposed = false, inputs = rows, outputs = cols)
            else -> fail()
        }
    }

    /** The encoder module plus the merged projection function. Returns the window row count and the function name. */
    private fun writeEncoder(encoder: File, out: File, adapter: File, prefill: File, dim: Int, projections: List<Projection>): Pair<Int, String> {
        val adp = adapter.useLines { lines -> lines.firstNotNullOfOrNull { SIG.matchEntire(it) } }
            ?: throw MaterializationException("cross-kv: no function signature in ${adapter.name}")
        val adpName = adp.groupValues[1].substringAfter("@")
        val window = MEMORY.matchEntire(adp.groupValues[3])?.takeIf { it.groupValues[2].toInt() == dim }?.groupValues?.get(1)?.toInt()
            ?: throw MaterializationException("cross-kv: ${adapter.name} returns ${adp.groupValues[3]}, not [1, C, $dim]")
        val function = if (adpName.endsWith("adapter")) adpName.removeSuffix("adapter") + "cross_kv" else "${adpName}_cross_kv"
        if (projections.any { it.inputs != dim }) throw MaterializationException("cross-kv: a K/V weight does not take $dim inputs")
        val width = projections.sumOf { it.outputs }

        // the weights side by side: merged[i][offset_j + o] = W_j[i][o]
        val merged = FloatArray(dim * width)
        val offsets = projections.runningFold(0) { a, p -> a + p.outputs }
        val byName = projections.withIndex().groupBy({ it.value.weight }, { it.index })
        prefill.bufferedReader().useLines { lines ->
            for (l in lines) {
                val d = DEF.find(if (l.length > LONG) l.substring(0, 256) else l) ?: continue
                val js = byName[d.groupValues[2]] ?: continue
                for (j in js) {
                    val p = projections[j]
                    val (r, c) = if (p.transposed) p.outputs to p.inputs else p.inputs to p.outputs
                    val w = constantValues(l, d.groupValues[2], r, c)
                    for (i in 0 until dim) for (o in 0 until p.outputs)
                        merged[i * width + offsets[j] + o] = if (p.transposed) w[o * dim + i] else w[i * p.outputs + o]
                }
            }
        }

        val text = encoder.readLines()
        val close = text.indexOfLast { it.trim() == "}" }
        if (close < 0 || text.none { SIG.matchEntire(it) != null }) throw MaterializationException("cross-kv: cannot read ${encoder.name}")
        val inT = "tensor<1x${window}x${dim}xf32>"
        val outT = "tensor<1x${window}x${width}xf32>"
        val wT = "tensor<${dim}x${width}xf32>"
        out.parentFile.mkdirs()
        out.bufferedWriter().use { w ->
            for (i in 0 until close) w.line(text[i])
            w.line("  func.func @$function(%arg0: $inT) -> ($outT) {")
            w.write("    %xkv_w = stablehlo.constant dense<[")
            for (i in 0 until dim) {
                if (i > 0) w.write(", ")
                w.write("[")
                for (o in 0 until width) { if (o > 0) w.write(", "); w.write(merged[i * width + o].toString()) }
                w.write("]")
            }
            w.line("]> : $wT")
            w.line("    %xkv = stablehlo.dot_general %arg0, %xkv_w, contracting_dims = [2] x [0] : ($inT, $wT) -> $outT")
            w.line("    return %xkv : $outT")
            w.line("  }")
            for (i in close until text.size) w.line(text[i])
        }
        return window to function
    }

    /** A `[rows, cols]` f32 constant's values, row-major: a nested literal or a splat. */
    private fun constantValues(line: String, name: String, rows: Int, cols: Int): FloatArray {
        val rhs = line.substringAfter(" = ")
        if (line.length <= LONG) {
            val m = CONSTANT.matchEntire(rhs) ?: throw MaterializationException("cross-kv: cannot read $name")
            if (m.groupValues[2].toInt() != rows || m.groupValues[3].toInt() != cols) throw MaterializationException("cross-kv: $name is not ${rows}x$cols")
            if (!m.groupValues[1].startsWith("[")) return FloatArray(rows * cols) { m.groupValues[1].trim().toFloat() }
        }
        if (!line.endsWith("]]> : tensor<${rows}x${cols}xf32>")) throw MaterializationException("cross-kv: $name is not a ${rows}x$cols f32 literal")
        return Int8WeightsRewrite.parseFloats(line, line.indexOf("[["), line.lastIndexOf("]]>") + 2, rows * cols, name, "cross-kv")
    }

    private fun operands(rhs: String): List<String> = NAME.findAll(rhs).map { it.value }.toList()

    private fun writePrefill(
        prefill: File, out: File, sig: MatchResult, ret: MatchResult, args: List<Pair<String, String>>, mem: String,
        cone: Set<String>, needed: Set<String>, defs: Map<String, String>,
        argOf: Map<String, String>, newArg: Map<String, String>, types: Map<String, String>,
    ) {
        // the weights only the cone uses go with it; one also used by the rest of prefill stays, with what it reads
        val keep = HashSet<String>()
        fun keepFrom(v: String) { if (v in needed && keep.add(v)) operands(defs.getValue(v)).forEach(::keepFrom) }
        defs.filterKeys { it !in cone && it !in needed }.values.flatMap(::operands).forEach(::keepFrom)
        operands(ret.groupValues[2]).forEach(::keepFrom)

        val retValues = ret.groupValues[2].split(", ")
        val retTypes = ret.groupValues[3].split(", ")
        val kept = retValues.indices.filter { retValues[it] !in cone }
        val newArgs = args.flatMap { (a, t) -> if (a == mem) newArg.map { (v, n) -> "$n: ${types.getValue(v)}" } else listOf("$a: $t") }
        val head = "${sig.groupValues[1]}(${newArgs.joinToString(", ")}) -> (${kept.joinToString(", ") { retTypes[it] }}) {"
        out.parentFile.mkdirs()
        out.bufferedWriter().use { w ->
            prefill.bufferedReader().useLines { lines ->
                for (l in lines) {
                    if (l.length > LONG) {
                        val name = DEF.find(l.substring(0, 256))?.groupValues?.get(2)
                        if (name == null || name !in needed || name in keep) w.line(l)
                        continue
                    }
                    if (SIG.matchEntire(l) != null && l.trimStart().startsWith(sig.groupValues[1].trimStart())) { w.line(head); continue }
                    val rt = RETURN.matchEntire(l)
                    if (rt != null) {
                        w.line("${rt.groupValues[1]}return ${kept.joinToString(", ") { retValues[it] }} : ${kept.joinToString(", ") { retTypes[it] }}")
                        continue
                    }
                    val d = DEF.matchEntire(l)
                    if (d != null && (d.groupValues[2] in cone || d.groupValues[2] in needed && d.groupValues[2] !in keep)) continue
                    w.line(if (argOf.isEmpty()) l else NAME.replace(l) { argOf[it.value] ?: it.value })
                }
            }
        }
    }

    private fun Writer.line(s: String) { write(s); write("\n") }
}
