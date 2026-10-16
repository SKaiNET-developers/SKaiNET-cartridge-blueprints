package sk.ainet.cartridge.detection.yolov8.export

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import onnx.ModelProto
import onnx.TensorProto
import pbandk.decodeFromByteArray
import sk.ainet.cartridge.detection.yolov8.model.YoloV8
import sk.ainet.cartridge.detection.yolov8.model.YoloV8Config
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/** The one entry point of the exported module; the runtime invokes it by this name. */
public const val FUNCTION: String = "yolov8_detect"

/**
 * Export step of the blueprint: the Ultralytics ONNX export's initializers (Conv weights, BatchNorm fused) baked
 * into the SKaiNET [YoloV8] graph, traced and lowered to StableHLO — one function `image[1,3,S,S] -> head[1,4+C,A]`,
 * weights folded as constants (IREE moves them into the parameter archive).
 *
 *   YOLO_ONNX=<yolov8n.onnx> YOLO_OUT=<dir>/yolov8n.mlir [YOLO_INPUT_SIZE=640] java … YoloV8ExportCliKt
 *
 * Only the ONNX initializers are read, never its graph. Class count, width and depth come from their shapes; every
 * parameter must be found with its exact shape, and every initializer must be used. Next to the `.mlir` goes
 * `manifest.json`: function, shapes, and a SHA-256 over the initializers (the ONNX file's own bytes are not
 * reproducible — the exporter stamps a date into its metadata).
 */
public fun main(args: Array<String>) {
    val onnx = File(System.getenv("YOLO_ONNX") ?: error("set YOLO_ONNX to the Ultralytics ONNX export (yolov8n.onnx)"))
    val out = File(System.getenv("YOLO_OUT") ?: args.getOrNull(0) ?: "build/mlir/yolov8n.mlir")
    val inputSize = System.getenv("YOLO_INPUT_SIZE")?.toInt() ?: 640

    val checkpoint = OnnxInitializers(onnx)
    val config = checkpoint.config(inputSize)
    log("${onnx.name}: ${checkpoint.names.size} initializers -> ${config.numClasses} classes, width ${config.width}, depth ${config.depth}, input $inputSize")

    val model = YoloV8(config)
    val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
    val baked = checkpoint.bake(model, ctx)
    val unused = checkpoint.names - baked
    check(unused.isEmpty()) { "initializers the graph did not use (graph and checkpoint disagree): $unused" }
    log("baked ${baked.size} parameters; every initializer was used")

    val image = Shape(1, 3, inputSize, inputSize)
    val graph = trace(ctx) { model.forward(void(image), it) }
    val mlir = StableHloConverterFactory.createExtended().convert(graph, FUNCTION).content
    check("stablehlo.constant" in mlir) { "the baked weights did not fold into constants" }
    val head = listOf(1, 4 + config.numClasses, config.anchors)
    val signature = "tensor<1x${4 + config.numClasses}x${config.anchors}xf32>"
    check(signature in mlir) { "the exported function does not return $signature" }

    out.absoluteFile.parentFile.mkdirs()
    out.writeText(mlir)
    log("${out.name}: @$FUNCTION, ${out.length() / 1024} KiB")

    val manifest = buildJsonObject {
        put("model", "yolov8")
        put("function", FUNCTION)
        put("input", buildJsonObject { put("name", "image"); put("dtype", "float32"); put("shape", JsonArray(image.dimensions.map(::JsonPrimitive))) })
        put("output", buildJsonObject { put("name", "head"); put("dtype", "float32"); put("shape", JsonArray(head.map(::JsonPrimitive))) })
        put("num_classes", config.numClasses); put("width", config.width); put("depth", config.depth); put("reg_max", config.regMax)
        put("parameters", baked.size)
        put("initializers_sha256", checkpoint.digest())
    }
    File(out.absoluteFile.parentFile, "manifest.json").writeText(Json { prettyPrint = true }.encodeToString(JsonObject.serializer(), manifest) + "\n")
}

private fun log(message: String) = println("[yolov8-export] $message")

/** The float initializers of an ONNX model, by name. */
internal class OnnxInitializers(file: File) {
    private val tensors: Map<String, TensorProto> =
        (ModelProto.decodeFromByteArray(file.readBytes()).graph ?: error("${file.name} has no graph")).initializer.associateBy { it.name }

    val names: Set<String> get() = tensors.keys

    fun dims(name: String): List<Int> = tensor(name).dims.map { it.toInt() }

    fun floats(name: String): FloatArray {
        val t = tensor(name)
        require(t.dataType == FLOAT) { "$name: data type ${t.dataType}, only float32 is supported" }
        val raw = t.rawData.array
        if (raw.isEmpty()) return t.floatData.toFloatArray()
        return FloatArray(raw.size / 4).also { ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
    }

    /** Geometry read off the checkpoint: classes from the class head, width from the stem, depth from stage 4. */
    fun config(inputSize: Int): YoloV8Config {
        val classes = dims("model.22.cv3.0.2.bias")[0]
        val width = dims("model.0.conv.weight")[0] / 64f
        val stage4 = names.mapNotNull { Regex("""model\.4\.m\.(\d+)\..*""").matchEntire(it)?.groupValues?.get(1)?.toInt() }.distinct().size
        return YoloV8Config(numClasses = classes, inputSize = inputSize, width = width, depth = stage4 / 6f)
    }

    /** Overwrites every parameter of [model] with the initializer of the same name; returns the names used. */
    fun bake(model: YoloV8, ctx: ExecutionContext): Set<String> {
        val used = LinkedHashSet<String>()
        for (p in model.parameters()) {
            val expected = p.value.shape.dimensions.toList()
            require(p.name in tensors) { "no initializer '${p.name}' for graph parameter ${expected}" }
            require(dims(p.name) == expected) { "'${p.name}': initializer ${dims(p.name)}, graph parameter $expected" }
            @Suppress("UNCHECKED_CAST")
            p.value = ctx.fromData(DenseFloatArrayTensorData<FP32>(Shape(*expected.toIntArray()), floats(p.name)) as TensorData<FP32, Float>, FP32::class)
            used += p.name
        }
        return used
    }

    /** SHA-256 over the initializers in name order: name, dims, little-endian float32 values. */
    fun digest(): String {
        val md = MessageDigest.getInstance("SHA-256")
        for (name in names.sorted()) {
            md.update(name.encodeToByteArray())
            md.update(dims(name).joinToString(",", "[", "]").encodeToByteArray())
            val values = floats(name)
            md.update(ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { asFloatBuffer().put(values) }.array())
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    private fun tensor(name: String) = tensors[name] ?: error("no initializer '$name'")

    private companion object {
        const val FLOAT = 1   // onnx.TensorProto.DataType.FLOAT
    }
}

private fun trace(ctx: DefaultGraphExecutionContext, body: (ExecutionContext) -> Unit): ComputeGraph {
    val tape = ctx.record {
        val current = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
        Execution.tapeStack.pushTape(current)
        try { body(this as ExecutionContext) } finally { Execution.tapeStack.popTape() }
    }.first
    return (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true, embedConstants = true)
}

private fun void(shape: Shape): VoidOpsTensor<FP32, Float> = VoidOpsTensor(
    object : TensorData<FP32, Float> {
        override val shape: Shape = shape
        override fun get(vararg indices: Int): Float = 0.0f
        override fun set(vararg indices: Int, value: Float) {}
    },
    FP32::class,
)
