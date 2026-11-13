package sk.ainet.cartridge.classification.moonshine.head

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * A materialized classification-head cartridge on disk (cartridge spec, ADR-012): `descriptor.json`,
 * `manifest.json` and `artifacts/<role>/…` as `materializeCartridge` wrote them. Locates things only — verifying
 * the manifest before load is the host's job, per the spec's threat model.
 */
public class PackDir(public val root: Path) {
    public constructor(root: String) : this(Path(root))

    init {
        require(SystemFileSystem.exists(Path(root, "descriptor.json"))) { "not a cartridge pack_dir (no descriptor.json): $root" }
    }

    public val descriptor: JsonObject by lazy { Json.parseToJsonElement(readText(Path(root, "descriptor.json"))).jsonObject }

    public val id: String get() = descriptor.getValue("id").jsonPrimitive.content

    /** `true` when the descriptor's target names an accelerator — the cartridge was built for a GPU. */
    public val targetsAccelerator: Boolean get() = descriptor["target"]?.jsonObject?.containsKey("accelerator") == true

    private val attributes: JsonObject get() = descriptor.getValue("attributes").jsonObject

    /** What the head expects to be fed: which encoder's rows, how many, how wide (descriptor `attributes.features`). */
    public val features: Features by lazy {
        val f = attributes.getValue("features").jsonObject
        Features(
            encoder = f["encoder"]?.jsonPrimitive?.content,
            revision = f["revision"]?.jsonPrimitive?.content,
            dim = f.getValue("dim").jsonPrimitive.int,
            maxRows = f.getValue("max_rows").jsonPrimitive.int,
        )
    }

    public data class Features(val encoder: String?, val revision: String?, val dim: Int, val maxRows: Int) {
        /** Floats in one input: `[maxRows, dim + 1]`. */
        val inputSize: Int get() = maxRows * (dim + 1)

        /**
         * The head's input from [count] encoder rows ([rows] is row-major `[count, dim]`): the first [maxRows] rows,
         * each followed by a 1 in the mask column; the rest stays zero.
         */
        public fun input(rows: FloatArray, count: Int): FloatArray {
            require(rows.size >= count * dim) { "$count rows of $dim need ${count * dim} floats, got ${rows.size}" }
            val out = FloatArray(inputSize)
            for (t in 0 until minOf(count, maxRows)) {
                rows.copyInto(out, t * (dim + 1), t * dim, (t + 1) * dim)
                out[t * (dim + 1) + dim] = 1f
            }
            return out
        }
    }

    /** The compiled head: one function, `rows[1, maxRows, dim + 1] -> logits[1, classes]`. */
    public val module: Path get() = file("artifacts/model/head.vmfb")

    /** The function to call in [module]. */
    public val function: String get() = attributes["runtime_hint"]?.jsonObject?.get("function")?.jsonPrimitive?.content ?: "classify"

    /** The fitted weights, for running the head eagerly with SKaiNET instead of the compiled module. */
    public val weights: Path get() = file("artifacts/weights/head.safetensors")

    public val labels: LabelSet by lazy { json.decodeFromString(LabelSet.serializer(), readText(file(LABEL_SET))) }

    public val calibration: Calibration by lazy { json.decodeFromString(Calibration.serializer(), readText(file(CALIBRATION))) }

    /** The pack's decision: its label set under its calibration. */
    public fun decision(): Decision = Decision(labels, calibration)

    private fun file(relative: String): Path = Path(root, relative).also {
        require(SystemFileSystem.exists(it)) { "pack_dir ${root.name} is missing $relative" }
    }

    public companion object {
        /** Where `classification/v1` puts the label set and the calibration in a pack. */
        public const val LABEL_SET: String = "artifacts/other/label-set.json"
        public const val CALIBRATION: String = "artifacts/calibration/calibration.json"

        private val json = Json { ignoreUnknownKeys = true }
        private fun readText(path: Path): String = SystemFileSystem.source(path).buffered().use { it.readString() }
    }
}
