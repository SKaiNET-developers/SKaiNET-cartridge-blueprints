package sk.ainet.cartridge.blueprint.steps

import sk.ainet.cartridge.blueprint.model.MaterializationException

/**
 * A `convert` step for the Synaptics Torq NPU's TOSA route: StableHLO → TOSA, compiled with
 * `--iree-input-type=tosa-torq`. That input pipeline ends with IREE's `MakeSingleDispatchForFunctionPass`, so a whole
 * function becomes ONE dispatch, as for the vendor's own models. The `stablehlo` route instead pays the Torq runtime's
 * per-dispatch bookkeeping (start/stop network, cache syncs) hundreds of times per call.
 *
 * The step has three parts; the middle one is StableHLO's own `--stablehlo-prepare-for-tosa
 * --stablehlo-legalize-to-tosa` (`stablehlo-opt`, run by [sk.ainet.cartridge.blueprint.gradle.TorqTosaTask] in the
 * `stablehlo-tosa` image), built from the Torq compiler's LLVM so its TOSA text parses there.
 *
 * 1. [prepareDivides], on the StableHLO: `x / broadcast(sqrt(z))` → `x * broadcast(rsqrt(z))` and `x / broadcast(y)` →
 *    `x * broadcast(1 / y)`. Legalized as is, the reciprocal runs on the broadcast tile (a layer norm's [T, D], a
 *    softmax's [H, T, T]) and `sqrt` becomes `pow`: neither has an NSS kernel, so both run as host code.
 * 2. the legalization.
 * 3. [finish], on its output: what the legalization leaves as StableHLO. `broadcast_in_dim` → `tosa.reshape` to the
 *    target rank, plus `tosa.tile` only when a consumer is not elementwise (TOSA elementwise ops broadcast
 *    implicitly); `sqrt` → `tosa.pow` 0.5; `rsqrt` → `tosa.rsqrt`; `dot_general` of a `[1, M, K]` activation with a
 *    `[K, N]` weight → `tosa.matmul` on the weight reshaped to `[1, K, N]`; batched `[1, H, M, K]` dot_generals
 *    (attention) → reshape (and transpose for `A·Bᵀ`) to `[H, ·, ·]`, `tosa.matmul`, reshape back. Any other
 *    StableHLO op fails the step.
 *
 * Compile the output with `backend = "torq-tosa"`, which adds [COMPILE_FLAGS]. Moonshine v2 tiny encoder (64 frames)
 * on an SL2610: one dispatch, 32 ms per call (stablehlo route with [TorqBf16Rewrite]: 137 ms, ~350 dispatches; A55
 * int8 × int8: 42 ms), golden transcripts identical.
 */
object TorqTosaRewrite {
    /**
     * Torq converts the graph to bf16 (`--torq-convert-dtypes`); CSS off (as fallback target it is ten times slower
     * than host code); no const-expr hoisting (hoisted weights end up as globals the single dispatch cannot load);
     * no elementwise fusion (the NSS runs single-op elementwise kernels only, a fused chain becomes host code);
     * weights inlined into the executable (otherwise they are cache-synced to the NPU on every call).
     */
    val COMPILE_FLAGS = listOf(
        "--torq-convert-dtypes",
        "--torq-disable-css",
        "--iree-opt-const-expr-hoisting=false",
        "--iree-dispatch-creation-enable-elementwise-fusion=false",
        "--iree-flow-inline-constants-max-byte-length=33554432",
    )

    /** Arguments of `stablehlo-opt` for the legalization (step 2). */
    val LEGALIZE_ARGS = listOf("--stablehlo-prepare-for-tosa", "--stablehlo-legalize-to-tosa")

    data class FinishResult(val mlir: String, val broadcasts: Int, val tiled: Int, val weightMatmuls: Int, val batchedMatmuls: Int, val roots: Int)

    private const val LONG = 4096
    private val NAME = Regex("""%[\w]+""")
    private val DEF = Regex("""^(\s*)(%[\w]+) = (.*)$""")
    private val SHLO_DEF = Regex("""^(\s*)(%[\w]+) = (stablehlo\.[a-z_]+)(.*)$""")

    // ---- 1. divides by a broadcast ---------------------------------------------------------------------------

    fun prepareDivides(mlir: String): Pair<String, Int> {
        val lines = mlir.split('\n')
        val defs = HashMap<String, Pair<String, String>>()  // value -> (op, rest of the line)
        for (l in lines) if (l.length <= LONG) SHLO_DEF.matchEntire(l)?.let { defs[it.groupValues[2]] = it.groupValues[3] to it.groupValues[4] }
        val out = ArrayList<String>(lines.size + 64)
        var n = 0
        for (l in lines) {
            val m = if (l.length <= LONG) SHLO_DEF.matchEntire(l) else null
            if (m != null && m.groupValues[3] == "stablehlo.divide") {
                val (ind, v, _, rest) = m.destructured
                val (a, b) = NAME.findAll(rest).map { it.value }.take(2).toList()
                val bcast = defs[b]
                if (bcast?.first == "stablehlo.broadcast_in_dim") {
                    val bc = bcast.second
                    val y = NAME.find(bc)!!.value
                    val dims = Regex("""dims = \[[^\]]*\]""").find(bc)!!.value
                    val (ty, tb) = Regex(""": \((tensor<[^>]+>)\) -> (tensor<[^>]+>)""").find(bc)!!.destructured
                    n++
                    val r = "%tq_r$n"
                    val src = defs[y]
                    if (src?.first == "stablehlo.sqrt") {
                        out += "$ind$r = stablehlo.rsqrt ${NAME.find(src.second)!!.value} : $ty"
                    } else {
                        out += "$ind%tq_one$n = stablehlo.constant dense<1.000000e+00> : $ty"
                        out += "$ind$r = stablehlo.divide %tq_one$n, $y : $ty"
                    }
                    out += "$ind%tq_b$n = stablehlo.broadcast_in_dim $r, $dims : ($ty) -> $tb"
                    out += "$ind$v = stablehlo.multiply $a, %tq_b$n : $tb"
                    continue
                }
            }
            out += l
        }
        return out.joinToString("\n") to n
    }

    // ---- 3. what the legalization leaves behind --------------------------------------------------------------

    private val BC = Regex("""^stablehlo\.broadcast_in_dim (%[\w]+), dims = \[([\d, ]*)\] : \((tensor<[^>]+>)\) -> (tensor<[^>]+>)$""")
    private val SQRT = Regex("""^stablehlo\.sqrt (%[\w]+) : (tensor<[^>]+>)$""")
    private val RSQRT = Regex("""^stablehlo\.rsqrt (%[\w]+) : (tensor<[^>]+>)$""")
    private val DOT = Regex(
        """^stablehlo\.dot_general (%[\w]+), (%[\w]+), (?:batching_dims = \[([\d, ]+)\] x \[([\d, ]+)\], )?contracting_dims = \[(\d)\] x \[(\d)\] : \((tensor<[^>]+>), (tensor<[^>]+>)\) -> (tensor<[^>]+>)$""",
    )
    private val ELEMENTWISE = listOf("tosa.add ", "tosa.sub ", "tosa.mul ", "tosa.maximum ", "tosa.minimum ", "tosa.pow ")

    private fun ty(d: List<Int>) = "tensor<${d.joinToString("x")}xf32>"
    private fun dims(t: String): List<Int> =
        if (t == "tensor<f32>") emptyList() else t.removePrefix("tensor<").removeSuffix("xf32>").split('x').map(String::toInt)

    fun finish(mlir: String): FinishResult {
        val lines = mlir.split('\n')
        val users = HashMap<String, MutableList<String>>()  // value -> right-hand sides that read it
        for (l in lines) if (l.length <= LONG) {
            val rhs = DEF.matchEntire(l)?.groupValues?.get(3) ?: l
            NAME.findAll(rhs).forEach { users.getOrPut(it.value, ::ArrayList) += rhs }
        }
        var fresh = 0
        fun next(p: String) = "%tq_$p${++fresh}"
        fun shape(ind: String, d: List<Int>, out: MutableList<String>): String {
            val s = next("shp")
            out += "$ind$s = tosa.const_shape {values = dense<[${d.joinToString(", ")}]> : tensor<${d.size}xindex>} : () -> !tosa.shape<${d.size}>"
            return s
        }
        fun reshape(ind: String, x: String, xt: String, d: List<Int>, out: MutableList<String>, into: String? = null): String {
            val s = shape(ind, d, out)
            val r = into ?: next("rs")
            out += "$ind$r = tosa.reshape $x, $s : ($xt, !tosa.shape<${d.size}>) -> ${ty(d)}"
            return r
        }

        val out = ArrayList<String>(lines.size + 512)
        val implicit = HashMap<String, Pair<String, String>>()  // broadcast result -> (rank-equal value, its type)
        var broadcasts = 0; var tiled = 0; var weightMm = 0; var batchedMm = 0; var roots = 0
        for (l in lines) {
            if (l.length <= LONG && l.trimStart().startsWith("func.func")) {  // every function: its own zero point
                out += l
                out += "    %tq_zp = \"tosa.const\"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>"
                continue
            }
            val m = if (l.length <= LONG) DEF.matchEntire(l) else null
            if (m == null || !m.groupValues[3].startsWith("stablehlo.")) { out += l; continue }
            val (ind, r, rhs) = m.destructured
            BC.matchEntire(rhs)?.let { b ->
                val (x, dd, at, bt) = b.destructured
                val a = dims(at); val target = dims(bt)
                val map = if (dd.isBlank()) emptyList() else dd.split(',').map { it.trim().toInt() }
                if (map != map.sorted()) throw MaterializationException("torq-tosa: broadcast_in_dim with permuted dims: `${l.trim().take(160)}`")
                val reshaped = MutableList(target.size) { 1 }.also { s -> map.forEachIndexed { i, d -> s[d] = a[i] } }
                var cur = x; var curT = at
                if (reshaped != a) { cur = reshape(ind, x, at, reshaped, out); curT = ty(reshaped) }
                val mult = target.zip(reshaped) { t, s -> t / s }
                val allElementwise = users[r].orEmpty().all { u -> ELEMENTWISE.any { u.trimStart().startsWith(it) } }
                when {
                    mult.all { it == 1 } -> reshape(ind, cur, curT, target, out, into = r)
                    !allElementwise -> {
                        val s = shape(ind, mult, out)
                        out += "$ind$r = tosa.tile $cur, $s : ($curT, !tosa.shape<${target.size}>) -> $bt"; tiled++
                    }
                    else -> {
                        val v = "%tq_bc" + r.drop(1)
                        out += "$ind$v = tosa.identity $cur : ($curT) -> $curT"
                        implicit[r] = v to curT
                    }
                }
                broadcasts++
                return@let
            } ?: SQRT.matchEntire(rhs)?.let { s ->
                val (x, t) = s.destructured
                val one = List(dims(t).size) { 1 }; val h = next("half")
                out += "$ind$h = \"tosa.const\"() <{values = dense<0.5> : ${ty(one)}}> : () -> ${ty(one)}"
                out += "$ind$r = tosa.pow $x, $h : ($t, ${ty(one)}) -> $t"; roots++
            } ?: RSQRT.matchEntire(rhs)?.let { s ->
                val (x, t) = s.destructured
                out += "$ind$r = tosa.rsqrt $x : ($t) -> $t"; roots++
            } ?: DOT.matchEntire(rhs)?.let { g ->
                val gv = g.groupValues
                val (a, w) = gv[1] to gv[2]
                val (ba, bb, ca, cb) = listOf(gv[3], gv[4], gv[5], gv[6])
                val (at, wt, rt) = Triple(gv[7], gv[8], gv[9])
                val ad = dims(at); val wd = dims(wt); val rd = dims(rt)
                when {
                    ba.isEmpty() && ad.size == 3 && wd.size == 2 && ca == "2" && cb == "0" && ad[0] == 1 -> {
                        val w3 = reshape(ind, w, wt, listOf(1) + wd, out)
                        out += "$ind$r = tosa.matmul $a, $w3, %tq_zp, %tq_zp : ($at, ${ty(listOf(1) + wd)}, tensor<1xf32>, tensor<1xf32>) -> $rt"
                        weightMm++
                    }
                    ba == "0, 1" && bb == "0, 1" && ad.size == 4 && ad[0] == 1 && ca == "3" && (cb == "3" || cb == "2") -> {
                        val h = ad[1]
                        val a3 = reshape(ind, a, at, ad.drop(1), out)
                        val (b3, b3t) = if (cb == "3") {  // A·Bᵀ: transpose B's last two dimensions first
                            val tt = ty(listOf(1, h, wd[3], wd[2])); val kt = next("kt")
                            out += "$ind$kt = tosa.transpose $w {perms = array<i32: 0, 1, 3, 2>} : ($wt) -> $tt"
                            reshape(ind, kt, tt, listOf(h, wd[3], wd[2]), out) to ty(listOf(h, wd[3], wd[2]))
                        } else reshape(ind, w, wt, wd.drop(1), out) to ty(wd.drop(1))
                        val o3 = ty(listOf(h, ad[2], rd[3])); val mm = next("bm")
                        out += "$ind$mm = tosa.matmul $a3, $b3, %tq_zp, %tq_zp : (${ty(ad.drop(1))}, $b3t, tensor<1xf32>, tensor<1xf32>) -> $o3"
                        reshape(ind, mm, o3, rd, out, into = r)
                        batchedMm++
                    }
                    else -> throw MaterializationException("torq-tosa: dot_general in a form this step does not translate: `${l.trim().take(200)}`")
                }
            } ?: throw MaterializationException("torq-tosa: the legalization left `${rhs.substringBefore(' ')}`, which this step does not translate: `${l.trim().take(160)}`")
        }
        // consumers of an implicitly broadcast value (elementwise only) read the rank-equal value instead
        val final = if (implicit.isEmpty()) out else out.map { l ->
            val m = if (l.length <= LONG) DEF.matchEntire(l) else null
            if (m == null || ELEMENTWISE.none { m.groupValues[3].startsWith(it) }) return@map l
            val (ind, res, rhs) = m.destructured
            val (ops, types) = rhs.split(" : ", limit = 2).let { it[0] to it[1] }
            val names = ops.substringAfter(' ').split(',').map(String::trim).toMutableList()
            if (names.none { it in implicit }) return@map l
            val inTypes = types.substring(1, types.indexOf(')')).split(", ").toMutableList()
            names.forEachIndexed { i, v -> implicit[v]?.let { (nv, t) -> names[i] = nv; inTypes[i] = t } }
            "$ind$res = ${ops.substringBefore(' ')} ${names.joinToString(", ")} : (${inTypes.joinToString(", ")})${types.substring(types.indexOf(')') + 1)}"
        }
        return FinishResult(final.joinToString("\n"), broadcasts, tiled, weightMm, batchedMm, roots)
    }
}
