package sk.ainet.cartridge.blueprint.steps

import sk.ainet.cartridge.blueprint.model.MaterializationException

/**
 * A `convert` step on exported StableHLO: the whole graph in bf16, shaped for the Synaptics Torq NPU (Torq 2.3.0).
 * The function keeps its f32 inputs and outputs; everything between them is bf16.
 *
 * 1. Weight matmuls: `dot_general %a, transpose(const [out, in])` reads a bf16 constant `[in, out]` instead (the
 *    transpose done here, the values rounded to nearest even). Torq lowers plain bf16 matmuls to the NPU but asserts on
 *    transposed-B ones, which is what IREE folds `transpose(const)` into; f32 matmuls run as slow host fallback.
 *    An im2col conv weight, `reshape(transpose(const [out, in, k], [0, 2, 1]))`, becomes its `[out, k·in]` bf16
 *    constant, so the transpose does not run on every call.
 * 2. Sum reductions over the last dimension become a matmul with an all-ones `[N, N]` matrix: the row sum, already
 *    broadcast to the input's shape. The ops between the reduce and its broadcast back (`/N`, `+eps`, `sqrt`,
 *    `reshape`, `broadcast_in_dim`) move to that full shape. In the full graph Torq gets the small row vectors in
 *    between (`[1, 64]`, `[1, 64, 1]`: layer-norm mean and variance, softmax sum) wrong; full-size ops are right.
 * 3. All types to bf16, with an optimization barrier on the result of every `dot_general` and of every transpose,
 *    so IREE fuses neither into a neighbour. Torq silently drops an elementwise op fused after a bf16
 *    matmul and ignores a transpose fused before one; a transpose fused with a bf16 elementwise op asserts. A
 *    `dot_general` that contracts the last dimension of both operands (attention Q·Kᵀ) gets an explicit transpose of
 *    its right operand behind a barrier, so it is no longer transposed-B. A strided slice (im2col of a strided conv)
 *    gets a barrier too: fused into the concatenate after it, the dispatch fails at run time (`xram_write` out of
 *    bounds).
 * 4. A barrier after every elementwise op that feeds another one: Torq runs an elementwise dispatch on the NPU only
 *    when it holds a single op, and fused chains (layer norm, GELU, residual adds) on the host. Also after one that
 *    feeds a transpose: a bf16 transpose fused with the multiply before it does not compile.
 *
 * Moonshine v2 tiny on an SL2610, measured against the f32 CPU graphs: encoder cosine 0.9995, 152 ms per 64-frame
 * call (179 ms without step 4); frontend cosine 0.99996, 48 ms per window (CPU: 185 ms). Both on the NPU: golden
 * transcripts identical, streaming RTF 0.57 (CPU, int8 weights: 0.77).
 * Compile the output with [COMPILE_FLAGS].
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
            "torq-bf16: no `dot_general` with a constant operand found — the module has no weight matmuls, " +
                "or the exporter's output format changed.",
        )
        val (lifted, ns) = liftSums(weights)
        val (bf16, nb) = toBf16(lifted)
        val (split, ne) = splitElementwise(bf16)
        return Result(split.joinToString("\n"), nw, ns, nb + ne)
    }

    private const val LONG = 4096  // longer lines are constant literals: no uses in them, never rewritten as ops
    private val NAME = Regex("""%[\w]+""")
    private fun b16(type: String) = type.replace("xf32>", "xbf16>").replace("tensor<f32>", "tensor<bf16>")

    // ---- 1. weight matmuls -------------------------------------------------------------------------------------

    private val CONST_HEAD = Regex("""^( *)(%[\w]+) = stablehlo\.constant dense<\[\[""")
    private val CONST_TAIL = Regex("""\]\]> : tensor<(\d+)x(\d+)xf32>$""")
    private val TRANSPOSE_2D = Regex("""^( *)(%[\w]+) = stablehlo\.transpose (%[\w]+), dims = \[1, 0\] : \(tensor<(\d+)x(\d+)xf32>\) -> tensor<\d+x\d+xf32>$""")
    private val TRANSPOSE_IM2COL = Regex("""^ *(%[\w]+) = stablehlo\.transpose (%[\w]+), dims = \[0, 2, 1\] : \(tensor<(\d+)x(\d+)x(\d+)xf32>\) -> tensor<[^>]+>$""")
    private val RESHAPE_2D = Regex("""^( *)(%[\w]+) = stablehlo\.reshape (%[\w]+) : \(tensor<[^>]+>\) -> tensor<(\d+)x(\d+)xf32>$""")
    private val ANY_DOT = Regex("""^ *%[\w]+ = stablehlo\.dot_general (%[\w]+), (%[\w]+),""")
    private val DOT_00 = Regex("""^( *)(%[\w]+) = stablehlo\.dot_general (%[\w]+), (%[\w]+), contracting_dims = \[(\d)\] x \[0\] : \((tensor<[^>]+xf32>), tensor<[^>]+xf32>\) -> (tensor<[^>]+xf32>)$""")

    private fun weightMatmuls(lines: List<String>): Pair<List<String>, Int> {
        val matrices = HashMap<String, String>()  // constant name -> its line
        for (l in lines) if (l.length > LONG) CONST_HEAD.find(l.substring(0, 256))?.let { matrices[it.groupValues[2]] = l }
        val weight = HashMap<String, Pair<String, String>>()  // transpose result -> bf16 constant name, its type
        val im2col = HashMap<String, Triple<String, Int, Int>>()  // transpose result -> conv weight, out, in (k = rest)
        val consts = HashSet(matrices.keys)  // operands that make a dot_general a weight matmul
        val out = ArrayList<String>(lines.size)
        var n = 0
        for (l in lines) {
            if (l.length > LONG) { out += l; continue }
            TRANSPOSE_IM2COL.matchEntire(l)?.let { m ->
                val (res, c, o, i) = m.destructured
                if (c in matrices) im2col[res] = Triple(c, o.toInt(), i.toInt())
            }
            val rs = RESHAPE_2D.matchEntire(l)
            val conv = rs?.let { im2col[it.groupValues[3]] }
            if (rs != null && conv != null) {
                val (ind, res, _, rows, cols) = rs.destructured
                val (c, o, i) = conv
                if (rows.toInt() != o || cols.toInt() % i != 0) throw MaterializationException("torq-bf16: cannot fold `${l.trim().take(140)}`")
                out += "$ind$res = stablehlo.constant dense<\"0x${im2colBf16Hex(matrices.getValue(c), c, o, i, cols.toInt() / i)}\"> : tensor<${rows}x${cols}xbf16>"
                consts += res
                continue
            }
            ANY_DOT.find(l)?.let { m -> if (m.groupValues[1] in consts || m.groupValues[2] in consts) n++ }
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
            } else out += l
            if (TRANSPOSE_2D.matchEntire(l)?.groupValues?.get(3) in matrices) consts += l.substringBefore(" =").trim()
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
        return bf16Hex(rows * cols) { n -> w[(n % rows) * cols + n / rows] }
    }

    /** The `[o, i, k]` f32 constant on [line] as the `[o, k·i]` bf16 hex literal im2col multiplies with. */
    private fun im2colBf16Hex(line: String, name: String, o: Int, i: Int, k: Int): String {
        val tail = Regex("""\]\]\]> : tensor<(\d+)x(\d+)x(\d+)xf32>$""").find(line, line.length - 128)
        if (tail == null || tail.groupValues.drop(1).map(String::toInt) != listOf(o, i, k))
            throw MaterializationException("torq-bf16: $name is not ${o}x${i}x$k")
        val w = Int8WeightsRewrite.parseFloats(line, line.indexOf("[[["), tail.range.first, o * i * k, name, "torq-bf16")
        return bf16Hex(o * i * k) { n -> val (r, c) = n / (k * i) to n % (k * i); w[(r * i + c % i) * k + c / i] }
    }

    private fun bf16Hex(count: Int, value: (Int) -> Float): String {
        val hex = "0123456789ABCDEF"
        return buildString(count * 4) {
            for (n in 0 until count) {
                val bits = java.lang.Float.floatToRawIntBits(value(n))
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
    private val TRANSPOSE_HI = Regex("""^( *)(%[\w]+) = stablehlo\.transpose %[\w]+, dims = \[\d+(?:, \d+)+\] : \(tensor<[^>]+>\) -> (tensor<[^>]+>)$""")
    private val DOT = Regex("""^( *)(%[\w]+) = stablehlo\.dot_general (%[\w]+), (%[\w]+), (.*)contracting_dims = \[(\d)\] x \[(\d)\] : \((tensor<[^>]+>), (tensor<[^>]+>)\) -> (tensor<[^>]+>)$""")
    private val STRIDED_SLICE = Regex("""^( *)(%[\w]+) = "stablehlo\.slice"\(.*strides = array<i64: ([\d, ]+)>\} : \(tensor<[^>]+>\) -> (tensor<[^>]+>)$""")
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
            val sl = STRIDED_SLICE.matchEntire(l)
            if (sl != null && sl.groupValues[3].split(", ").any { it != "1" }) {
                val (ind, r, _, t) = sl.destructured
                out += l.replaceFirst("$r = ", "${r}_s = ")
                out += "$ind$r = stablehlo.optimization_barrier ${r}_s : $t"; n++
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

    // ---- 4. one elementwise op per dispatch ------------------------------------------------------------------

    private val ELEMENTWISE_OPS = setOf("add", "subtract", "multiply", "divide", "maximum", "negate", "sqrt", "rsqrt", "tanh", "exponential")
    private val ELEMENTWISE = Regex("""^( *)(%[\w]+) = stablehlo\.(${ELEMENTWISE_OPS.joinToString("|")}) .* (tensor<[^>]+>)$""")
    private val PASS_THROUGH = setOf("reshape", "broadcast_in_dim")  // IREE fuses elementwise ops across these
    private val OP = Regex("""^ *(%[\w]+) = stablehlo\.(\w+)""")

    /**
     * A barrier after every elementwise op whose result feeds another elementwise op (directly or through a reshape
     * or broadcast), so no dispatch holds two of them: Torq runs single-op bf16 elementwise dispatches on the NPU and
     * fused ones on the host (`x * y + z` fused: host, 1.7 ms; split: two NPU dispatches, 0.3 ms each).
     */
    private fun splitElementwise(lines: List<String>): Pair<List<String>, Int> {
        val op = HashMap<String, String>()
        val users = HashMap<String, MutableList<String>>()
        for (l in lines) if (l.length <= LONG) OP.find(l)?.let { m ->
            op[m.groupValues[1]] = m.groupValues[2]
            NAME.findAll(l.substringAfter(" = ")).map { it.value }.toSet().forEach { users.getOrPut(it, ::ArrayList) += m.groupValues[1] }
        }
        fun feedsElementwise(v: String): Boolean = users[v].orEmpty().any { u ->
            val o = op[u]
            o in ELEMENTWISE_OPS || o == "transpose" || o in PASS_THROUGH && feedsElementwise(u)
        }
        val out = ArrayList<String>(lines.size + 200)
        var n = 0
        for (l in lines) {
            val m = if (l.length <= LONG) ELEMENTWISE.matchEntire(l) else null
            if (m != null && feedsElementwise(m.groupValues[2])) {
                val (ind, r, _, t) = m.destructured
                out += l.replaceFirst("$r = ", "${r}_e = ")
                out += "$ind$r = stablehlo.optimization_barrier ${r}_e : $t"; n++
            } else out += l
        }
        return out to n
    }
}
