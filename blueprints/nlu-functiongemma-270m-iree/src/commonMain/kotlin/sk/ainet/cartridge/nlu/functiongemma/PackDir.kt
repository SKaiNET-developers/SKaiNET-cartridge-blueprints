package sk.ainet.cartridge.nlu.functiongemma

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * The materialized cartridge on disk (cartridge spec, ADR-012): `descriptor.json`, `manifest.json` and
 * `artifacts/<role>/…` exactly as `materializeCartridge` wrote them. This class only *locates* things;
 * verifying the manifest before load is the host's (or its staging step's) job, per the spec's threat model.
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

    public fun graph(name: String): Path = file("artifacts/model/gemma-$name.vmfb")
    public fun parameters(name: String): Path = file("artifacts/weights/gemma-$name.irpa")
    public val tokenizerGguf: Path get() = file("artifacts/tokenizer/functiongemma-270m-it-Q8_0.gguf")
    public val toolCatalog: Path get() = file("artifacts/other/tool-catalog.json")

    private fun file(relative: String): Path = Path(root, relative).also {
        require(SystemFileSystem.exists(it)) { "pack_dir ${root.name} is missing $relative" }
    }

    internal companion object {
        fun readText(path: Path): String = SystemFileSystem.source(path).buffered().use { it.readString() }
    }
}
