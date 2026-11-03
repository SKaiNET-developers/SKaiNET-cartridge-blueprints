package sk.ainet.cartridge.blueprint.steps

import sk.ainet.cartridge.blueprint.model.MaterializationException
import java.io.File
import java.io.Writer

/**
 * A `convert` step on the exported streaming decoder: the cross-attention K/V of each memory row is computed once, by
 * the adapter that produces the row, instead of by every prefill over the whole padded memory.
 *
 * The prefill graph projects its memory argument (`[1, M, D]`, M = 256 padded rows) to every layer's cross K and V
 * (`[1, H, M, d]`) with per-row matmuls, reshapes and transposes: the "memory cone". Those ops see one row at a time,
 * so a row's K/V never changes once the row is final, yet the streaming loop runs prefill on every restart and hop,
 * and the cone is ~93% of its work (Moonshine v2 tiny: 629 of 674 MFLOP).
 *
 * - prefill: the memory argument is replaced by the cone's outputs, one argument per distinct value, in the order the
 *   graph returned them (layer 0 K, layer 0 V, …); the cone, its weights and those outputs are dropped.
 * - adapter: the cone is appended after the adapter's output (`[1, C, D]`, C = window rows), its types retyped from
 *   M to C rows and its constants renamed `%kv_…`; the adapter returns its output, then the K/V of its C rows.
 *
 * The caller keeps a `[H, M, d]` cache per layer and side, copies the finalized rows of each window in, and passes the
 * cache to prefill and the decoder steps. The decoder-step graph is unchanged. Streams line by line: an exported
 * prefill is ~300 MB of text, single constants several MB.
 */
object CrossKvCacheRewrite {
    data class Result(val outputs: Int, val memoryRows: Int, val windowRows: Int, val coneOps: Int)

    private const val LONG = 4096  // longer lines are constant literals: no uses in them
    private val NAME = Regex("""%[\w]+""")
    private val DEF = Regex("""^( *)(%[\w]+) = (.*)$""")
    private val SIG = Regex("""^( *func\.func @[\w]+)\((.*)\) -> \((.*)\) \{$""")
    private val ARG = Regex("""(%arg\d+): (tensor<[^>]+>)""")
    private val RETURN = Regex("""^( *)return (.+) : (.+)$""")
    private val MEMORY = Regex("""tensor<1x(\d+)x(\d+)xf32>""")

    fun rewrite(prefill: File, adapter: File, prefillOut: File, adapterOut: File): Result {
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

        // the static values the cone needs (weights and their transposes), for the adapter
        val needed = LinkedHashSet<String>()
        fun need(v: String) { if (v in static && needed.add(v)) operands(defs.getValue(v)).forEach(::need) }
        cone.forEach { c -> operands(defs.getValue(c)).forEach(::need) }

        val window = writeAdapter(adapter, adapterOut, prefill, mem, rows, dim.toInt(), cone, needed, defs, boundary)
        writePrefill(prefill, prefillOut, s, r, args, mem, cone, needed, defs, argOf, newArg, boundaryType)
        return Result(boundary.size, rows, window, cone.size)
    }

    private fun operands(rhs: String): List<String> = NAME.findAll(rhs).map { it.value }.toList()

    /** The result type of an op's right-hand side: after `->`, or after the last `:` for same-type ops. */
    private fun resultType(rhs: String): String = if ("-> " in rhs) rhs.substringAfterLast("-> ") else rhs.substringAfterLast(": ")

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

    /** Appends the cone to the adapter, retyped to its window rows. Returns the window row count. */
    private fun writeAdapter(
        adapter: File, out: File, prefill: File, mem: String, rows: Int, dim: Int,
        cone: Set<String>, needed: Set<String>, defs: Map<String, String>, boundary: List<String>,
    ): Int {
        val text = adapter.readLines()
        val sigAt = text.indexOfFirst { SIG.matchEntire(it) != null }
        val retAt = text.indexOfLast { RETURN.matchEntire(it) != null }
        if (sigAt < 0 || retAt < 0) throw MaterializationException("cross-kv: cannot read ${adapter.name}")
        val sig = SIG.matchEntire(text[sigAt])!!
        val ret = RETURN.matchEntire(text[retAt])!!
        if (", " in ret.groupValues[2]) throw MaterializationException("cross-kv: ${adapter.name} already returns more than one value")
        val outType = ret.groupValues[3]
        val window = MEMORY.matchEntire(outType)?.takeIf { it.groupValues[2].toInt() == dim }?.groupValues?.get(1)?.toInt()
            ?: throw MaterializationException("cross-kv: ${adapter.name} returns $outType, not [1, C, $dim]")
        val rename = (needed + cone).associateWith { "%kv_" + it.drop(1) } + (mem to ret.groupValues[2])
        fun renamed(l: String) = NAME.replace(l) { rename[it.value] ?: it.value }
        // only the memory-derived types change rows; a weight with M rows would make that ambiguous
        val rowDim = Regex("""(?<=[<x])$rows(?=x)""")
        needed.forEach { v ->
            if (defs.getValue(v) != "stablehlo.constant" && rowDim.containsMatchIn(defs.getValue(v)))
                throw MaterializationException("cross-kv: $v has a dimension of $rows, the memory's row count")
        }
        fun retyped(rhs: String) = Regex("""tensor<[^>]+>""").replace(rhs) { t ->
            if (rowDim.findAll(t.value).count() > 1) throw MaterializationException("cross-kv: ${t.value} has more than one dimension of $rows")
            rowDim.replace(t.value, "$window")
        }

        val ind = ret.groupValues[1]
        val kvTypes = boundary.map { v -> resultType(retyped(defs.getValue(v))) }
        out.parentFile.mkdirs()
        out.bufferedWriter().use { w ->
            for (i in 0 until retAt) w.line(if (i == sigAt) "${sig.groupValues[1]}(${sig.groupValues[2]}) -> ($outType, ${kvTypes.joinToString(", ")}) {" else text[i])
            // weights (and their transposes) from prefill, in prefill order
            prefill.bufferedReader().useLines { lines ->
                for (l in lines) {
                    val name = DEF.find(if (l.length > LONG) l.substring(0, 256) else l)?.groupValues?.get(2) ?: continue
                    if (name !in needed) continue
                    w.line(if (l.length > LONG) l.replaceFirst(name, rename.getValue(name)) else renamed(l))
                }
            }
            for (c in cone) w.line("$ind${rename.getValue(c)} = ${retyped(renamed(defs.getValue(c)))}")
            w.line("${ind}return ${ret.groupValues[2]}, ${boundary.joinToString(", ") { rename.getValue(it) }} : $outType, ${kvTypes.joinToString(", ")}")
            for (i in retAt + 1 until text.size) w.line(text[i])
        }
        return window
    }

    private fun Writer.line(s: String) { write(s); write("\n") }
}
