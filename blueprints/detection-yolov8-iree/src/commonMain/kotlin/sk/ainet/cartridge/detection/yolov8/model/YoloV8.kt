package sk.ainet.cartridge.detection.yolov8.model

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Conv2d
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.types.FP32
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Geometry of an Ultralytics YOLOv8 detection model. The defaults are YOLOv8n (width 0.25, depth 0.33) on COCO.
 */
public data class YoloV8Config(
    val numClasses: Int = 80,
    /** Square input side; a multiple of the largest stride (32). */
    val inputSize: Int = 640,
    val width: Float = 0.25f,
    val depth: Float = 0.33f,
    val maxChannels: Int = 1024,
    /** Bins of the distribution-focal-loss box regression. */
    val regMax: Int = 16,
) {
    init {
        require(inputSize > 0 && inputSize % STRIDES.last() == 0) { "inputSize must be a positive multiple of ${STRIDES.last()}" }
    }

    /** Ultralytics `make_divisible(min(c, max_channels) * width, 8)`. */
    public fun channels(c: Int): Int = (ceil(min(c, maxChannels) * width / 8.0).toInt() * 8)

    /** Ultralytics `max(round(n * depth), 1)`. */
    public fun repeats(n: Int): Int = max((n * depth).roundToInt(), 1)

    /** Anchor points over all strides: 8400 at 640. */
    public val anchors: Int get() = STRIDES.sumOf { (inputSize / it) * (inputSize / it) }

    public companion object {
        public val STRIDES: List<Int> = listOf(8, 16, 32)
    }
}

/**
 * YOLOv8 detection in the SKaiNET DSL: letterboxed RGB `[1, 3, S, S]` in `[0, 1]` → decoded head
 * `[1, 4 + classes, anchors]` — rows 0–3 box center x, center y, width, height in input pixels, then per-class
 * sigmoid scores. This is the export-mode output of the Ultralytics `Detect` head (DFL expectation, anchor-point
 * decode, stride scaling), so the graph is the whole tensor function of the cartridge; NMS stays on the host.
 *
 * Layers and parameters carry the names of the Ultralytics checkpoint (`model.0.conv.weight`, …,
 * `model.22.dfl.conv.weight`), BatchNorm already fused into the convolutions, as the Ultralytics ONNX export writes
 * them. Parameters start as shape-only placeholders; [parameters] lists them for baking.
 */
public class YoloV8(public val config: YoloV8Config = YoloV8Config()) : Module<FP32, Float>() {
    override val name: String = "yolov8"

    private val c = config::channels
    private val n = config::repeats

    private val l0 = ConvBlock("model.0", 3, c(64), 3, 2)
    private val l1 = ConvBlock("model.1", c(64), c(128), 3, 2)
    private val l2 = C2f("model.2", c(128), c(128), n(3), shortcut = true)
    private val l3 = ConvBlock("model.3", c(128), c(256), 3, 2)
    private val l4 = C2f("model.4", c(256), c(256), n(6), shortcut = true)
    private val l5 = ConvBlock("model.5", c(256), c(512), 3, 2)
    private val l6 = C2f("model.6", c(512), c(512), n(6), shortcut = true)
    private val l7 = ConvBlock("model.7", c(512), c(1024), 3, 2)
    private val l8 = C2f("model.8", c(1024), c(1024), n(3), shortcut = true)
    private val l9 = Sppf("model.9", c(1024), c(1024))
    // 10 upsample, 11 concat(10, 6)
    private val l12 = C2f("model.12", c(1024) + c(512), c(512), n(3), shortcut = false)
    // 13 upsample, 14 concat(13, 4)
    private val l15 = C2f("model.15", c(512) + c(256), c(256), n(3), shortcut = false)
    private val l16 = ConvBlock("model.16", c(256), c(256), 3, 2)
    // 17 concat(16, 12)
    private val l18 = C2f("model.18", c(256) + c(512), c(512), n(3), shortcut = false)
    private val l19 = ConvBlock("model.19", c(512), c(512), 3, 2)
    // 20 concat(19, 9)
    private val l21 = C2f("model.21", c(512) + c(1024), c(1024), n(3), shortcut = false)
    private val detect = Detect("model.22", listOf(c(256), c(512), c(1024)), config)

    override val modules: List<Module<FP32, Float>> =
        listOf(l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l12, l15, l16, l18, l19, l21, detect)

    /** Every parameter of the graph, depth-first, with its checkpoint name. */
    public fun parameters(): List<ModuleParameter<FP32, Float>> = collect(this, ArrayList())

    override fun onForward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val ops = ctx.ops
        val p3b = l4.forward(l3.forward(l2.forward(l1.forward(l0.forward(input, ctx), ctx), ctx), ctx), ctx)  // 1/8
        val p4b = l6.forward(l5.forward(p3b, ctx), ctx)                                                         // 1/16
        val p5 = l9.forward(l8.forward(l7.forward(p4b, ctx), ctx), ctx)                                         // 1/32
        val h12 = l12.forward(ops.concat(listOf(up(p5, ctx), p4b), dim = 1), ctx)
        val p3 = l15.forward(ops.concat(listOf(up(h12, ctx), p3b), dim = 1), ctx)
        val p4 = l18.forward(ops.concat(listOf(l16.forward(p3, ctx), h12), dim = 1), ctx)
        val p5d = l21.forward(ops.concat(listOf(l19.forward(p4, ctx), p5), dim = 1), ctx)
        return detect.heads(listOf(p3, p4, p5d), ctx)
    }

    private fun up(x: Tensor<FP32, Float>, ctx: ExecutionContext) = ctx.ops.upsample2d(x, 2 to 2, UpsampleMode.Nearest)

    private fun collect(m: Module<FP32, Float>, out: MutableList<ModuleParameter<FP32, Float>>): List<ModuleParameter<FP32, Float>> {
        @Suppress("UNCHECKED_CAST")
        if (m is ModuleParameters<*, *>) out += (m as ModuleParameters<FP32, Float>).params
        for (child in m.modules) collect(child, out)
        return out
    }
}

/** A shape-only placeholder; baking replaces the parameter's value. */
internal fun placeholder(vararg dims: Int): Tensor<FP32, Float> = VoidOpsTensor(
    object : TensorData<FP32, Float> {
        override val shape: Shape = Shape(*dims)
        override fun get(vararg indices: Int): Float = 0.0f
        override fun set(vararg indices: Int, value: Float) {}
    },
    FP32::class,
)

internal fun conv(name: String, cin: Int, cout: Int, k: Int, s: Int = 1, bias: Boolean = true): Conv2d<FP32, Float> =
    Conv2d(
        inChannels = cin, outChannels = cout, kernelSize = k to k, stride = s to s, padding = k / 2 to k / 2,
        bias = bias, name = name, initWeights = placeholder(cout, cin, k, k), initBias = if (bias) placeholder(cout) else null,
    )

/** Ultralytics `Conv`: conv (BatchNorm fused) → SiLU. */
internal class ConvBlock(override val name: String, cin: Int, cout: Int, k: Int, s: Int = 1) : Module<FP32, Float>() {
    private val conv = conv("$name.conv", cin, cout, k, s)
    override val modules: List<Module<FP32, Float>> = listOf(conv)
    override fun onForward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> =
        ctx.ops.silu(conv.forward(input, ctx))
}

/** Ultralytics `Bottleneck` with `e = 1.0`, 3x3 / 3x3, optional residual. */
internal class Bottleneck(override val name: String, ch: Int, private val shortcut: Boolean) : Module<FP32, Float>() {
    private val cv1 = ConvBlock("$name.cv1", ch, ch, 3)
    private val cv2 = ConvBlock("$name.cv2", ch, ch, 3)
    override val modules: List<Module<FP32, Float>> = listOf(cv1, cv2)
    override fun onForward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val y = cv2.forward(cv1.forward(input, ctx), ctx)
        return if (shortcut) ctx.ops.add(input, y) else y
    }
}

/** Ultralytics `C2f`: 1x1 → split in two → n bottlenecks on the last chunk → concat all → 1x1. */
internal class C2f(override val name: String, cin: Int, cout: Int, n: Int, shortcut: Boolean) : Module<FP32, Float>() {
    private val hidden = cout / 2
    private val cv1 = ConvBlock("$name.cv1", cin, 2 * hidden, 1)
    private val m = List(n) { Bottleneck("$name.m.$it", hidden, shortcut) }
    private val cv2 = ConvBlock("$name.cv2", (2 + n) * hidden, cout, 1)
    override val modules: List<Module<FP32, Float>> = listOf(cv1) + m + cv2
    override fun onForward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val y = ctx.ops.split(cv1.forward(input, ctx), splitSize = hidden, dim = 1).toMutableList()
        for (b in m) y += b.forward(y.last(), ctx)
        return cv2.forward(ctx.ops.concat(y, dim = 1), ctx)
    }
}

/** Ultralytics `SPPF`: 1x1 → three chained 5x5 max-pools → concat → 1x1. */
internal class Sppf(override val name: String, cin: Int, cout: Int) : Module<FP32, Float>() {
    private val cv1 = ConvBlock("$name.cv1", cin, cin / 2, 1)
    private val cv2 = ConvBlock("$name.cv2", cin / 2 * 4, cout, 1)
    override val modules: List<Module<FP32, Float>> = listOf(cv1, cv2)
    override fun onForward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val x = cv1.forward(input, ctx)
        val p1 = ctx.ops.maxPool2d(x, 5 to 5, 1 to 1, 2 to 2)
        val p2 = ctx.ops.maxPool2d(p1, 5 to 5, 1 to 1, 2 to 2)
        val p3 = ctx.ops.maxPool2d(p2, 5 to 5, 1 to 1, 2 to 2)
        return cv2.forward(ctx.ops.concat(listOf(x, p1, p2, p3), dim = 1), ctx)
    }
}

/** Two 3x3 `Conv` blocks and a plain 1x1 conv: one box (`cv2.i`) or class (`cv3.i`) branch of the head. */
internal class HeadBranch(override val name: String, cin: Int, mid: Int, cout: Int) : Module<FP32, Float>() {
    private val b0 = ConvBlock("$name.0", cin, mid, 3)
    private val b1 = ConvBlock("$name.1", mid, mid, 3)
    private val b2 = conv("$name.2", mid, cout, 1)
    override val modules: List<Module<FP32, Float>> = listOf(b0, b1, b2)
    override fun onForward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> =
        b2.forward(b1.forward(b0.forward(input, ctx), ctx), ctx)
}

/**
 * Ultralytics `Detect` in export mode: per level, box distribution (`cv2.i`, 4 x regMax) and class logits
 * (`cv3.i`); flattened and concatenated over levels; DFL expectation (softmax over the bins, then the fixed
 * `dfl.conv` projection `0..regMax-1`); anchor-point decode to center/size; stride scaling; sigmoid on classes.
 */
internal class Detect(override val name: String, ch: List<Int>, private val config: YoloV8Config) : Module<FP32, Float>() {
    private val nc = config.numClasses
    private val regMax = config.regMax
    private val boxMid = maxOf(16, ch[0] / 4, regMax * 4)
    private val clsMid = maxOf(ch[0], min(nc, 100))
    private val box = ch.mapIndexed { i, c -> HeadBranch("$name.cv2.$i", c, boxMid, 4 * regMax) }
    private val cls = ch.mapIndexed { i, c -> HeadBranch("$name.cv3.$i", c, clsMid, nc) }
    private val dfl = conv("$name.dfl.conv", regMax, 1, 1, bias = false)
    override val modules: List<Module<FP32, Float>> = box + cls + dfl

    private val anchorTables: AnchorTables by lazy { AnchorTables.of(config) }

    fun heads(features: List<Tensor<FP32, Float>>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val ops = ctx.ops
        val a = config.anchors
        val flat = { t: Tensor<FP32, Float>, rows: Int -> ops.reshape(t, Shape(1, rows, t.shape[2] * t.shape[3])) }
        val boxes = ops.concat(features.mapIndexed { i, f -> flat(box[i].forward(f, ctx), 4 * regMax) }, dim = 2)  // [1, 4R, A]
        val scores = ops.concat(features.mapIndexed { i, f -> flat(cls[i].forward(f, ctx), nc) }, dim = 2)        // [1, nc, A]

        // DFL: [1, 4R, A] → [1, 4, R, A] → [1, R, 4, A] → softmax over R → 1x1 conv R→1 → [1, 4, A] distances.
        val bins = ops.softmax(ops.permute(ops.reshape(boxes, Shape(1, 4, regMax, a)), intArrayOf(0, 2, 1, 3)), dim = 1)
        val dist = ops.reshape(dfl.forward(bins, ctx), Shape(1, 4, a))
        val (lt, rb) = ops.split(dist, splitSize = 2, dim = 1)

        // center = (anchor + (rb - lt) / 2) * stride, size = (lt + rb) * stride
        val t = anchorTables
        val center = ops.add(table(t.anchorPixels, a, ctx), ops.multiply(ops.subtract(rb, lt), table(t.halfStride, a, ctx)))
        val size = ops.multiply(ops.add(lt, rb), table(t.stride, a, ctx))
        return ops.concat(listOf(center, size, ops.sigmoid(scores)), dim = 1)
    }

    override fun onForward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> =
        error("Detect takes the three feature maps; call heads()")

    private fun table(values: FloatArray, a: Int, ctx: ExecutionContext): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(1, 2, a), FP32::class, values)
}

/**
 * Per-anchor constants of the decode, `[1, 2, anchors]` row-major (x row, then y row), levels in stride order and
 * each level row-major over its grid — the order the head's flattened feature maps have.
 */
internal class AnchorTables(val anchorPixels: FloatArray, val halfStride: FloatArray, val stride: FloatArray) {
    companion object {
        fun of(config: YoloV8Config): AnchorTables {
            val a = config.anchors
            val anchor = FloatArray(2 * a); val half = FloatArray(2 * a); val full = FloatArray(2 * a)
            var i = 0
            for (s in YoloV8Config.STRIDES) {
                val g = config.inputSize / s
                for (y in 0 until g) for (x in 0 until g) {
                    anchor[i] = (x + 0.5f) * s; anchor[a + i] = (y + 0.5f) * s
                    half[i] = s / 2f; half[a + i] = s / 2f
                    full[i] = s.toFloat(); full[a + i] = s.toFloat()
                    i++
                }
            }
            return AnchorTables(anchor, half, full)
        }
    }
}
