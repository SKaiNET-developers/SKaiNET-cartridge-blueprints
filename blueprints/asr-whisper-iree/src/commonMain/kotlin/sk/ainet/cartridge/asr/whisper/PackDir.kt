package sk.ainet.cartridge.asr.whisper

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * A materialized Whisper cartridge on disk (cartridge spec, ADR-012): `descriptor.json`, `manifest.json` and
 * `artifacts/<role>/<flavor>/…` as `materializeCartridge` wrote them. Locates things only — verifying the manifest
 * before load is the host's job, per the spec's threat model.
 *
 * Flavors are languages. A pack materialized with both holds `artifacts/model/en/…` and `artifacts/model/de/…`;
 * [languages] lists what is there.
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

    /** The languages this pack was materialized with (descriptor `attributes.languages`). */
    public val languages: List<String>
        get() = descriptor["attributes"]?.jsonObject?.get("languages")?.jsonArray?.map { it.jsonPrimitive.content } ?: emptyList()

    /** The model files of [language], by role directory: three graphs, the shared parameter archive, contract and tokenizer. */
    public fun model(language: String): ModelFiles = ModelFiles(
        encoderVmfb = file("artifacts/model/$language/encoder.vmfb"),
        prefillVmfb = file("artifacts/model/$language/prefill.vmfb"),
        stepVmfb = file("artifacts/model/$language/step.vmfb"),
        paramsIrpa = file("artifacts/weights/$language/params.irpa"),
        contractJson = file("artifacts/other/$language/whisper-contract.json"),
        tokenizerJson = file("artifacts/tokenizer/$language/tokenizer.json"),
    )

    /** The graph contract of [language]: function names, shapes and the decode prompt (`whisper-contract.json`). */
    public fun contract(language: String): Contract = Contract.parse(readText(model(language).contractJson))

    public data class ModelFiles(
        val encoderVmfb: Path, val prefillVmfb: Path, val stepVmfb: Path,
        val paramsIrpa: Path, val contractJson: Path, val tokenizerJson: Path,
    ) {
        public val all: List<Path> get() = listOf(encoderVmfb, prefillVmfb, stepVmfb, paramsIrpa, contractJson, tokenizerJson)
    }

    /**
     * What a runtime needs from the exporter's manifest: the three function names, the parameter scope the prefill
     * and step modules reference, the decoder geometry and the prompt to prefill (`<sot> <lang> <transcribe>
     * <notimestamps>`, or `<sot> <notimestamps>` for an English-only checkpoint).
     */
    public data class Contract(
        val version: Int,
        val encoderFunction: String, val prefillFunction: String, val stepFunction: String,
        val parameterScope: String,
        val nLayers: Int, val dModel: Int, val audioCtx: Int, val melFrames: Int, val maxPositions: Int,
        val vocab: Int, val eot: Int, val promptIds: List<Int>,
    ) {
        public companion object {
            public fun parse(json: String): Contract {
                val o = Json.parseToJsonElement(json).jsonObject
                fun int(key: String): Int = o.getValue(key).jsonPrimitive.int
                val functions = o.getValue("functions").jsonObject
                fun fn(key: String): String = functions.getValue(key).jsonPrimitive.content
                return Contract(
                    version = int("contractVersion"),
                    encoderFunction = fn("encoder"), prefillFunction = fn("prefill"), stepFunction = fn("step"),
                    parameterScope = o.getValue("parameterScope").jsonPrimitive.content,
                    nLayers = int("nLayers"), dModel = int("dModel"), audioCtx = int("audioCtx"), melFrames = int("melFrames"),
                    maxPositions = int("maxPositions"), vocab = int("vocab"), eot = int("eot"),
                    promptIds = o.getValue("promptIds").jsonArray.map { it.jsonPrimitive.int },
                ).also { require(it.version == 2) { "whisper contract version ${it.version}; this API reads version 2" } }
            }
        }
    }

    private fun file(relative: String): Path = Path(root, relative).also {
        require(SystemFileSystem.exists(it)) { "pack_dir ${root.name} is missing $relative" }
    }

    internal companion object {
        fun readText(path: Path): String = SystemFileSystem.source(path).buffered().use { it.readString() }
    }
}
