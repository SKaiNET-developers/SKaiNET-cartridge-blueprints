package sk.ainet.cartridge.classification.moonshine.head

import android.content.Context
import kotlinx.io.files.Path
import sk.ainet.transformers.iree.android.FloatTensor
import sk.ainet.transformers.iree.android.IreeModuleRunner
import java.io.File

/**
 * The Android [HeadRunner]: the pack's compiled head on SKaiNET-transformers' generic single-shot runner
 * (`IreeModuleRunner`, `libskainet_iree_module.so`). The module carries its weights. Creating it loads the module
 * on the device.
 */
public class IreeHeadRunner(
    pack: PackDir,
    /** `IreeModuleRunner.VULKAN_DEVICE` or `IreeModuleRunner.DEFAULT_DEVICE` (CPU). Defaults to what the pack was built for. */
    device: String = if (pack.targetsAccelerator) IreeModuleRunner.VULKAN_DEVICE else IreeModuleRunner.DEFAULT_DEVICE,
) : HeadRunner {
    private val shape = intArrayOf(1, pack.features.maxRows, pack.features.dim + 1)
    private val runner = IreeModuleRunner(pack.module.toString(), null, pack.function, device)

    override fun run(input: FloatArray): FloatArray = runner.run(FloatTensor(input, shape)).data

    override fun close(): Unit = runner.close()
}

/** Opening a materialized pack as a [Classifier] on Android. */
public object ClassificationHeadCartridge {
    /** The pack's classifier over [IreeHeadRunner]; close it to release the device. */
    public fun open(pack: PackDir, device: String? = null): Classifier =
        Classifier(pack, if (device == null) IreeHeadRunner(pack) else IreeHeadRunner(pack, device))

    /**
     * Copies a pack shipped under `assets/<assetPath>/` into [into] (once; existing non-empty files are kept) and
     * returns it as a [PackDir]. The IREE runtime opens files by path, so assets must be on the file system.
     */
    public fun packFromAssets(context: Context, assetPath: String, into: File = File(context.filesDir, assetPath)): PackDir {
        val assets = context.assets
        fun copy(rel: String) {
            val children = assets.list(rel).orEmpty()
            if (children.isEmpty()) {
                val out = File(into, rel.removePrefix(assetPath).trimStart('/'))
                if (out.exists() && out.length() > 0) return
                out.parentFile?.mkdirs()
                assets.open(rel).use { i -> out.outputStream().use { i.copyTo(it) } }
            } else children.forEach { copy("$rel/$it") }
        }
        copy(assetPath)
        return PackDir(Path(into.absolutePath))
    }
}
