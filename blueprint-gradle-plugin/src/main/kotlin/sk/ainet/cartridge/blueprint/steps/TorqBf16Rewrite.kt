package sk.ainet.cartridge.blueprint.steps

import sk.ainet.cartridge.blueprint.model.MaterializationException

/**
 * A `convert` step on exported StableHLO: the whole graph in bf16, shaped for the Synaptics Torq NPU (Torq 2.3.0).
 * The function keeps its f32 inputs and outputs; everything between them is bf16.
 *
 * 1. Weight matmuls: `dot_general %a, transpose(const [out, in])` reads a bf16 constant `[in, out]` instead (the
 *    transpose done here, the values rounded to nearest even). Torq lowers plain bf16 matmuls to the NPU but asserts on
 *    transposed-B ones, which is what IREE folds `transpose(const)` into; f32 matmuls run as slow host fallback.
 * 2. Sum reductions over the last dimension become a matmul with an all-ones `[N, N]` matrix: the row sum, already
 *    broadcast to the input's shape. The ops between the reduce and its broadcast back (`/N`, `+eps`, `sqrt`,
 *    `reshape`, `broadcast_in_dim`) move to that full shape. In the full graph Torq gets the small row vectors in
 *    between (`[1, 64]`, `[1, 64, 1]`: layer-norm mean and variance, softmax sum) wrong; full-size ops are right.
 * 3. All types to bf16, with an optimization barrier on the result of every `dot_general` and of every transpose of
 *    rank 3 or more, so IREE fuses neither into a neighbour. Torq silently drops an elementwise op fused after a bf16
 *    matmul and ignores a transpose fused before one; a transpose fused with a bf16 elementwise op asserts. A
 *    `dot_general` that contracts the last dimension of both operands (attention Q·Kᵀ) gets an explicit transpose of
 *    its right operand behind a barrier, so it is no longer transposed-B.
 *
 * Moonshine v2 tiny encoder on an SL2610, measured against its f32 CPU graph: cosine 0.9995, golden transcripts
 * identical, streaming RTF 0.74 (CPU, int8 weights: 0.77). Compile the output with [COMPILE_FLAGS].
 *
 * Text-level on purpose, like [HostGatherRewrite]: reviewable next to the exported module.
 */
object TorqBf16Rewrite {
    /**
     * Extra `iree-compile` flags for the rewritten module on Torq 2.3.0: the NPU's compute subsystem off (it runs
     * this graph, but ten times slower), no shared transient buffers between dispatches (the runtime crashes on
     * them), and the weights inlined into the executables so they are not synced to the NPU on every call.
     */
    val COMPILE_FLAGS = listOf(
        "--torq-disable-css",
        "--iree-scheduling-optimize-bindings=false",
        "--iree-flow-inline-constants-max-byte-length=4194304",
    )

    data class Result(val mlir: String, val weightMatmuls: Int, val sumsAsMatmuls: Int, val barriers: Int)

    fun rewrite(mlir: String): Result {
        val (weights, nw) = weightMatmuls(mlir.split('\n'))
        if (nw == 0) throw MaterializationException(
            "torq-bf16: no `dot_general %a, transpose(constant)` found — the module has no weight matmuls, " +
                "or the exporter's output format changed.",
        )
        val (lifted, ns) = liftSums(weights)
        val (bf16, nb) = toBf16(lifted)
        return Result(bf16.joinToString("\n"), nw, ns, nb)
    }

    private const val LONG = 4096  // longer lines are constant literals: no uses in them, never rewritten as ops
    private val NAME = Regex("""%[\w]+""")
    private fun b16(type: String) = type.replace("xf32>", "xbf16>").replace("tensor<f32>", "tensor<bf16>")

    // ---- 1. weight matmuls -------------------------------------------------------------------------------------

    private val CONST_HEAD = Regex("""^( *)(%[\w]+) = stablehlo\.constant dense<\[\[""")
    private val CONST_TAIL = Regex("""\]\]> : tensor<(\d+)x(\d+)xf32>$""")
    private val TRANSPOSE_2D = Regex("""^( *)(%[\w]+) = stablehlo\.transpose (%[\w]+), dims = \[1, 0\] : \(tensor<(\d+)x(\d+)xf32>\) -> tensor<\d+x\d+xf32>$""")
    private val DOT_00 = Regex("""^( *)(%[\w]+) = stablehlo\.dot_general (%[\w]+), (%[\w]+), contracting_dims = \[(\d)\] x \[0\] : \((tensor<[^>]+xf32>), tensor<[^>]+xf32>\) -> (tensor<[^>]+xf32>)$""")

    private fun weightMatmuls(lines: List<String>): Pair<List<String>, Int> {
        val matrices = HashMap<String, String>()  // constant name -> its line
        for (l in lines) if (l.length > LONG) CONST_HEAD.find(l.substring(0, 256))?.let { matrices[it.groupValues[2]] = l }
        val weight = HashMap<String, Pair<String, String>>()  // transpose result -> bf16 constant name, its type
        val out = ArrayList<String>(lines.size)
        var n = 0
        for (l in lines) {
            if (l.length > LONG) { out += l; continue }
            TRANSPOSE_2D.matchEntire(l)?.let { m ->
                val (ind, res, c, r, k) = m.destructured
                matrices[c]?.let { line ->
                    val type = "tensor<${k}x${r}xbf16>"
                    out += "$ind${res}_w16 = stablehlo.constant dense<\"0x${transposedBf16Hex(line, c, r.toInt(), k.toInt())}\"> : $type"
                    weight[res] = "${res}_w16" to type
                }
            }
            val dot = DOT_00.matchEntire(l)
            val w = dot?.let { weight[it.groupValues[4]] }
            if (dot != null && w != null) {
                val (ind, res, a, _, cd, ta, tr) = dot.destructured
                out += "$ind$res = stablehlo.dot_general $a, ${w.first}, contracting_dims = [$cd] x [0] : ($ta, ${w.second}) -> $tr"
                n++
            } else out += l
        }
        return dropUnused(out) to n
    }

    /** The `[rows, cols]` f32 constant on [line] as a `[cols, rows]` bf16 hex literal (little-endian). */
    private fun transposedBf16Hex(line: String, name: String, rows: Int, cols: Int): String {
        val tail = CONST_TAIL.find(line, line.length - 128)
            ?: throw MaterializationException("torq-bf16: cannot read the type of $name")
        if (tail.groupValues[1].toInt() != rows || tail.groupValues[2].toInt() != cols)
            throw MaterializationException("torq-bf16: $name is not ${rows}x$cols")
        val w = Int8WeightsRewrite.parseFloats(line, line.indexOf("[["), tail.range.first, rows * cols, name, "torq-bf16")
        val hex = "0123456789ABCDEF"
        return buildString(rows * cols * 4) {
            for (j in 0 until cols) for (i in 0 until rows) {
                val bits = java.lang.Float.floatToRawIntBits(w[i * cols + j])
                val h = ((bits + 0x7FFF + ((bits ushr 16) and 1)) ushr 16) and 0xFFFF  // round to nearest even
                for (b in intArrayOf(h and 0xFF, h ushr 8)) { append(hex[b ushr 4]); append(hex[b and 0xF]) }
            }
        }
    }

    /** Drops constants and transposes nobody uses any more (the f32 weights and their transposes). */
    private fun dropUnused(lines: List<String>): List<String> {
        var cur = lines
        repeat(2) {
            val uses = HashMap<String, Int>()
            for (l in cur) if (l.length <= LONG) {
                val rhs = l.substringAfter(" = ", l)
                NAME.findAll(rhs).forEach { uses.merge(it.value, 1, Int::plus) }
            }
            cur = cur.filter { l ->
                val m = Regex("""^ *(%[\w]+) = stablehlo\.(constant|transpose) """).find(l.substring(0, minOf(l.length, 200)))
                m == null || uses.containsKey(m.groupValues[1])
            }
        }
        return cur
    }

    // ---- 2. sums as ones-matmuls -------------------------------------------------------------------------------

    private val REDUCE_ADD = Regex("""^( *)(%[\w]+) = stablehlo\.reduce\((%[\w]+) init: %[\w]+\) applies stablehlo\.add across dimensions = \[(\d)\] : \((tensor<([0-9x]+)x(\d+)xf32>), tensor<f32>\) -> tensor<[^>]+>$""")
    private val SHAPE_OP = Regex("""^( *)(%[\w]+) = stablehlo\.(reshape|broadcast_in_dim) (%[\w]+)[ ,].*?(tensor<[^>]+>)$""")
    private val BINARY = Regex("""^( *)(%[\w]+) = stablehlo\.(add|subtract|multiply|divide) (%[\w]+), (%[\w]+) : (tensor<[^>]+>)$""")
    private val UNARY = Regex("""^( *)(%[\w]+) = stablehlo\.(sqrt|rsqrt|exponential) (%[\w]+) : (tensor<[^>]+>)$""")
    private val SPLAT = Regex("""^ *(%[\w]+) = stablehlo\.constant dense<([^\[">]+)> : tensor<""")

    private fun liftSums(lines: List<String>): Pair<List<String>, Int> {
        val splats = HashMap<String, String>()
        for (l in lines) if (l.length <= LONG) SPLAT.find(l)?.let { splats[it.groupValues[1]] = it.groupValues[2] }
        val full = HashMap<String, String>()   // lifted value -> its full type
        val alias = HashMap<String, String>()  // dropped reshape / broadcast -> the lifted value it stands for
        val sizes = sortedSetOf<Int>()
        val out = ArrayList<String>(lines.size)
        var n = 0
        fun sub(v: String) = alias[v] ?: v
        for (line in lines) {
            if (line.length > LONG) { out += line; continue }
            val red = REDUCE_ADD.matchEntire(line)
            if (red != null && red.groupValues[4].toInt() == red.groupValues[6].count { it == 'x' } + 1) {
                val (ind, r, a, d, t, _, k) = red.destructured
                out += "$ind$r = stablehlo.dot_general ${sub(a)}, %ones$k, contracting_dims = [$d] x [0] : ($t, tensor<${k}x${k}xf32>) -> $t"
                full[r] = t; sizes += k.toInt(); n++
                continue
            }
            val shape = SHAPE_OP.matchEntire(line)
            if (shape != null && shape.groupValues[4] in full) {
                val (_, r, op, a, tr) = shape.destructured
                if (op == "broadcast_in_dim" && tr != full[a])
                    throw MaterializationException("torq-bf16: cannot lift `${line.trim().take(140)}`")
                alias[r] = sub(a); full[r] = full.getValue(a)
                continue
            }
            val bin = BINARY.matchEntire(line)
            if (bin != null && (bin.groupValues[4] in full || bin.groupValues[5] in full)) {
                val (ind, r, op, a, b, t) = bin.destructured
                val ft = full[a] ?: full.getValue(b)
                val operands = listOf(a, b).map { v ->
                    val s = splats[v]
                    if (v !in full && s != null) {
                        out += "$ind${r}_${v.drop(1)}c = stablehlo.constant dense<$s> : $ft"
                        "${r}_${v.drop(1)}c"
                    } else sub(v)
                }
                out += "$ind$r = stablehlo.$op ${operands[0]}, ${operands[1]} : $ft"
                if (t != ft) full[r] = ft
                continue
            }
            val un = UNARY.matchEntire(line)
            if (un != null && un.groupValues[4] in full) {
                val (ind, r, op, a) = un.destructured
                out += "$ind$r = stablehlo.$op ${sub(a)} : ${full.getValue(a)}"
                full[r] = full.getValue(a)
                continue
            }
            val l = if (alias.isEmpty()) line else NAME.replace(line) { alias[it.value] ?: it.value }
            val rhs = l.substringAfter(" = ", "")
            NAME.findAll(rhs).firstOrNull { it.value in full && full.getValue(it.value) !in rhs }?.let {
                throw MaterializationException("torq-bf16: ${it.value} is a row sum used by `${l.trim().take(140)}`, which cannot be lifted")
            }
            out += l
        }
        val sig = out.indexOfFirst { it.trimStart().startsWith("func.func") }
        out.addAll(sig + 1, sizes.map { "    %ones$it = stablehlo.constant dense<1.0> : tensor<${it}x${it}xf32>" })
        return out to n
    }

    // ---- 3. bf16 + barriers ------------------------------------------------------------------------------------

    private val ARG = Regex("""(%arg\d+): (tensor<[^>]+xf32>)""")
    private val RETURN = Regex("""^( *)return (.+) : (.+)$""")
    private val TRANSPOSE_HI = Regex("""^( *)(%[\w]+) = stablehlo\.transpose %[\w]+, dims = \[\d+(?:, \d+){2,}\] : \(tensor<[^>]+>\) -> (tensor<[^>]+>)$""")
    private val DOT = Regex("""^( *)(%[\w]+) = stablehlo\.dot_general (%[\w]+), (%[\w]+), (.*)contracting_dims = \[(\d)\] x \[(\d)\] : \((tensor<[^>]+>), (tensor<[^>]+>)\) -> (tensor<[^>]+>)$""")
    private val HEX_SPLAT = Regex("""dense<0x([0-9A-Fa-f]{8})>""")

    private fun toBf16(lines: List<String>): Pair<List<String>, Int> {
        val sig = lines.indexOfFirst { it.trimStart().startsWith("func.func") }
        val args = ARG.findAll(lines[sig]).map { it.groupValues[1] to it.groupValues[2] }.toList()
        val argNames = args.associate { (a, _) -> a to "${a}_16" }
        val out = ArrayList<String>(lines.size + 400)
        out += lines.subList(0, sig + 1)
        for ((a, t) in args) out += "    ${a}_16 = stablehlo.convert $a : ($t) -> ${b16(t)}"
        var n = 0
        for (raw in lines.subList(sig + 1, lines.size)) {
            if (raw.trimStart().startsWith("//")) { out += raw; continue }
            val ret = RETURN.matchEntire(raw)
            if (ret != null) {
                val (ind, values, types) = ret.destructured
                val vt = values.split(", ").zip(types.split(", "))
                for ((v, t) in vt) if (t != b16(t)) out += "$ind${v}_out = stablehlo.convert $v : (${b16(t)}) -> $t"
                out += "${ind}return ${vt.joinToString { (v, t) -> if (t != b16(t)) "${v}_out" else v }} : $types"
                continue
            }
            if (raw.contains("dense<\"0x") && raw.contains("xf32>"))
                throw MaterializationException("torq-bf16: hex f32 literal not supported: `${raw.trim().take(80)}…`")
            if (raw.length > LONG) { out += b16(raw); continue }
            val l = HEX_SPLAT.replace(b16(NAME.replace(raw) { argNames[it.value] ?: it.value })) { m ->
                val bits = m.groupValues[1].toLong(16).toInt()
                "dense<0x%04X>".format(((bits + 0x7FFF + ((bits ushr 16) and 1)) ushr 16) and 0xFFFF)
            }
            val tr = TRANSPOSE_HI.matchEntire(l)
            if (tr != null) {
                val (ind, r, t) = tr.destructured
                out += l.replaceFirst("$r = ", "${r}_t = ")
                out += "$ind$r = stablehlo.optimization_barrier ${r}_t : $t"; n++
                continue
            }
            val dot = DOT.matchEntire(l)
            if (dot != null) {
                val (ind, r, a, b0, batch, ca, cb0, ta, tb0, tr0) = dot.destructured
                var b = b0; var cb = cb0; var tb = tb0
                val rankA = ta.count { it == 'x' }; val rankB = tb.count { it == 'x' }
                if (ca.toInt() == rankA - 1 && cb.toInt() == rankB - 1 && rankB >= 2) {  // transposed-B: A·Bᵀ
                    val d = tb.removePrefix("tensor<").removeSuffix(">").split('x')
                    val perm = (0 until rankB).map { if (it == rankB - 1) rankB - 2 else if (it == rankB - 2) rankB - 1 else it }
                    val tbt = "tensor<${(d.dropLast(3) + d[rankB - 1] + d[rankB - 2] + d.last()).joinToString("x")}>"
                    out += "$ind${r}_kt = stablehlo.transpose $b, dims = $perm : ($tb) -> $tbt"
                    out += "$ind${r}_ktb = stablehlo.optimization_barrier ${r}_kt : $tbt"
                    b = "${r}_ktb"; tb = tbt; cb = "${rankB - 2}"; n++
                }
                out += "$ind${r}_o = stablehlo.dot_general $a, $b, ${batch}contracting_dims = [$ca] x [$cb] : ($ta, $tb) -> $tr0"
                out += "$ind$r = stablehlo.optimization_barrier ${r}_o : $tr0"; n++
                continue
            }
            out += l
        }
        return out to n
    }
}
