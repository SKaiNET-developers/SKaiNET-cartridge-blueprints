package sk.ainet.cartridge.nlu.qwen

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
 * The materialized cartridge on disk (cartridge spec, ADR-012): `descriptor.json`, `manifest.json` and
 * `artifacts/<role>/…` exactly as `materializeCartridge` wrote them, with one sub-directory per flavor for everything
 * a flavor owns (graphs, parameter archives, the qwen-kv-v1 manifest, the tokenizer). This class only *locates*
 * things; verifying the manifest before load is the host's (or its staging step's) job, per the spec's threat model.
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

    /** The flavors this pack bundles, by blueprint flavor id (`qwen3-600m`), in descriptor order. */
    public val flavors: List<String> get() = flavorEntries().map { it.first }

    /** The chat template a flavor was built for (`attributes.chat_template` of its descriptor entry). */
    public fun chatTemplate(flavor: String): QwenTemplate {
        val entry = flavorEntries().firstOrNull { it.first == flavor }?.second ?: error("pack_dir ${root.name} has no flavor '$flavor'")
        val name = entry["attributes"]?.jsonObject?.get("chat_template")?.jsonPrimitive?.content
            ?: error("flavor '$flavor' names no attributes.chat_template")
        return QwenTemplate.byName(name)
    }

    public fun graph(flavor: String, name: String): Path = file("artifacts/model/$flavor/qwen-$name.vmfb")
    public fun parameters(flavor: String, name: String): Path = file("artifacts/weights/$flavor/qwen-$name.irpa")
    public fun kvManifest(flavor: String): Path = file("artifacts/other/$flavor/manifest.json")
    public fun tokenizerGguf(flavor: String): Path {
        val dir = Path(root, "artifacts/tokenizer/$flavor")
        val gguf = SystemFileSystem.list(dir).singleOrNull { it.name.endsWith(".gguf") }
            ?: error("pack_dir ${root.name} has no single .gguf under artifacts/tokenizer/$flavor")
        return gguf
    }
    public val toolCatalog: Path get() = file("artifacts/other/tool-catalog.json")

    /** (flavor id, descriptor entry); a flavor entry's id is `<cartridge id>-<flavor id>`. */
    private fun flavorEntries(): List<Pair<String, JsonObject>> {
        val prefix = "$id-"
        return descriptor["flavors"]?.jsonArray.orEmpty().map { it.jsonObject }.map { e ->
            val full = e.getValue("id").jsonPrimitive.content
            require(full.startsWith(prefix)) { "flavor id '$full' does not extend the cartridge id '$id'" }
            full.removePrefix(prefix) to e
        }
    }

    private fun file(relative: String): Path = Path(root, relative).also {
        require(SystemFileSystem.exists(it)) { "pack_dir ${root.name} is missing $relative" }
    }

    internal companion object {
        fun readText(path: Path): String = SystemFileSystem.source(path).buffered().use { it.readString() }
    }
}
