package sk.ainet.cartridge.detection.yolov8

import android.content.Context
import kotlinx.io.files.Path
import sk.ainet.cartridge.detection.yolov8.model.YoloV8
import sk.ainet.cartridge.profile.yolo.v1.YoloEngine
import sk.ainet.transformers.iree.android.FloatTensor
import sk.ainet.transformers.iree.android.IreeModuleRunner
import java.io.File

/**
 * The Android [HeadRunner]: the pack's compiled graph and parameter archive on SKaiNET-transformers' generic
 * single-shot runner (`IreeModuleRunner`, `libskainet_iree_module.so`). Creating it loads the module on the device.
 */
public class IreeHeadRunner(
    pack: PackDir,
    /** `IreeModuleRunner.VULKAN_DEVICE` or `IreeModuleRunner.DEFAULT_DEVICE` (CPU). Defaults to what the pack was built for. */
    device: String = if (pack.targetsAccelerator) IreeModuleRunner.VULKAN_DEVICE else IreeModuleRunner.DEFAULT_DEVICE,
) : HeadRunner {
    private val size = pack.settings.inputResolution
    private val shape = intArrayOf(1, 3, size, size)
    private val runner: IreeModuleRunner = pack.model().let { files ->
        IreeModuleRunner(files.vmfb.toString(), files.paramsIrpa.toString(), YoloV8.FUNCTION, device)
    }

    override fun run(input: FloatArray): FloatArray = runner.run(FloatTensor(input, shape)).data

    override fun close(): Unit = runner.close()
}

/** Opening a materialized pack as a `yolo/v1` engine on Android. */
public object YoloV8Cartridge {
    /** The pack's engine over [IreeHeadRunner]; close the engine to release the device. */
    public fun open(pack: PackDir, device: String? = null): YoloEngine =
        YoloV8Engine(pack, if (device == null) IreeHeadRunner(pack) else IreeHeadRunner(pack, device))

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
