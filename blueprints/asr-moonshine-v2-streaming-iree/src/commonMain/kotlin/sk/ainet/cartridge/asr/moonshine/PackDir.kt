package sk.ainet.cartridge.asr.moonshine

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * A materialized Moonshine cartridge on disk (cartridge spec, ADR-012): `descriptor.json`, `manifest.json` and
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

    /** The eight model files of [language], by role directory: five graphs, the parameter archive, the two tables. */
    public fun model(language: String): ModelFiles = ModelFiles(
        frontendVmfb = file("artifacts/model/$language/frontend.vmfb"),
        encoderVmfb = file("artifacts/model/$language/encoder.vmfb"),
        adapterVmfb = file("artifacts/model/$language/adapter.vmfb"),
        prefillVmfb = file("artifacts/model/$language/prefill.vmfb"),
        stepVmfb = file("artifacts/model/$language/step.vmfb"),
        paramsIrpa = file("artifacts/weights/$language/params.irpa"),
        vocabBin = file("artifacts/tokenizer/$language/vocab.bin"),
        decEmbedBin = file("artifacts/weights/$language/dec_embed.bin"),
    )

    public data class ModelFiles(
        val frontendVmfb: Path, val encoderVmfb: Path, val adapterVmfb: Path, val prefillVmfb: Path, val stepVmfb: Path,
        val paramsIrpa: Path, val vocabBin: Path, val decEmbedBin: Path,
    ) {
        public val all: List<Path> get() = listOf(frontendVmfb, encoderVmfb, adapterVmfb, prefillVmfb, stepVmfb, paramsIrpa, vocabBin, decEmbedBin)
    }

    private fun file(relative: String): Path = Path(root, relative).also {
        require(SystemFileSystem.exists(it)) { "pack_dir ${root.name} is missing $relative" }
    }

    internal companion object {
        fun readText(path: Path): String = SystemFileSystem.source(path).buffered().use { it.readString() }
    }
}
