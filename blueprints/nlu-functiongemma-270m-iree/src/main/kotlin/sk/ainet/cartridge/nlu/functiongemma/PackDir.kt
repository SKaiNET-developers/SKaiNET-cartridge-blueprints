package sk.ainet.cartridge.nlu.functiongemma

import android.content.Context
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File

/**
 * The materialized cartridge on disk (cartridge spec, ADR-012): `descriptor.json`, `manifest.json` and
 * `artifacts/<role>/…` exactly as `materializeCartridge` wrote them. This class only *locates* things;
 * verifying the manifest before load is the host's (or its staging step's) job, per the spec's threat model.
 */
public class PackDir(public val root: File) {
    init {
        require(File(root, "descriptor.json").isFile) { "not a cartridge pack_dir (no descriptor.json): ${root.path}" }
    }

    public val descriptor: JsonObject by lazy { Json.parseToJsonElement(File(root, "descriptor.json").readText()).jsonObject }

    public val id: String get() = descriptor.getValue("id").jsonPrimitive.content

    /** `vulkan` when the descriptor's target names an accelerator, else the CPU device. */
    public val device: String get() = if (descriptor["target"]?.jsonObject?.containsKey("accelerator") == true) "vulkan" else "local-task"

    public fun graph(name: String): File = file("artifacts/model/gemma-$name.vmfb")
    public fun parameters(name: String): File = file("artifacts/weights/gemma-$name.irpa")
    public val tokenizerGguf: File get() = file("artifacts/tokenizer/functiongemma-270m-it-Q8_0.gguf")
    public val toolCatalog: File get() = file("artifacts/other/tool-catalog.json")

    private fun file(path: String): File = File(root, path).also {
        require(it.isFile) { "pack_dir ${root.name} is missing $path" }
    }

    public companion object {
        /**
         * For apps that ship the pack_dir inside their APK assets: copies `assetPath/` to [into] once (IREE opens
         * files by path; the parameter archives are mmapped) and returns the [PackDir]. Files whose size already
         * matches are skipped. Apps that provision the pack_dir another way construct [PackDir] directly.
         */
        public fun fromAssets(context: Context, assetPath: String, into: File = File(context.filesDir, assetPath)): PackDir {
            copyTree(context, assetPath, into)
            return PackDir(into)
        }

        private fun copyTree(context: Context, assetPath: String, target: File) {
            val children = context.assets.list(assetPath).orEmpty()
            if (children.isEmpty()) {
                val size = runCatching { context.assets.openFd(assetPath).use { it.length } }.getOrNull()
                if (size != null && target.isFile && target.length() == size) return
                target.parentFile?.mkdirs()
                context.assets.open(assetPath).use { input -> target.outputStream().use { input.copyTo(it, 1 shl 20) } }
                return
            }
            target.mkdirs()
            for (child in children) copyTree(context, "$assetPath/$child", File(target, child))
        }
    }
}
