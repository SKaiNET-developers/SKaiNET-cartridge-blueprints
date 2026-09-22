package sk.ainet.cartridge.asr.moonshine

import android.content.Context
import kotlinx.io.files.Path
import sk.ainet.transformers.iree.android.IreeMoonshineStream
import java.io.File

/**
 * The Android binding: [StreamingAsrCartridge] over SKaiNET-transformers' released streaming runtime
 * (`IreeMoonshineStream`, `libskainet_moonshine_stream.so`) and the model files of one language flavor of a
 * materialized [PackDir].
 *
 * One session at a time; calls on the session are synchronized. The engine is created lazily on the first
 * [open] and lives until [close] — creating it loads the five graphs on the device (seconds on a Mali GPU).
 */
public class MoonshineAsrCartridge(
    public val pack: PackDir,
    /** BCP-47 tag; the flavor is chosen with [Language.select] against what the pack holds. */
    languageTag: String = "en",
    /** `IreeMoonshineStream.VULKAN_DEVICE` (GPU) or `IreeMoonshineStream.CPU_DEVICE`. Defaults to what the pack was built for. */
    private val device: String = if (pack.targetsAccelerator) IreeMoonshineStream.VULKAN_DEVICE else IreeMoonshineStream.CPU_DEVICE,
) : StreamingAsrCartridge {

    override val id: String = pack.id
    override val language: String = Language.select(languageTag, pack.languages)
        ?: throw IllegalArgumentException("pack ${pack.id} holds languages ${pack.languages}, none serves '$languageTag'")

    private var stream: IreeMoonshineStream? = null
    private var session: Session? = null

    @Synchronized
    override fun open(): StreamingAsrSession {
        session?.let { return it }
        val files = pack.model(language)
        val s = IreeMoonshineStream(
            device,
            IreeMoonshineStream.Files(
                files.frontendVmfb.toString(), files.encoderVmfb.toString(), files.adapterVmfb.toString(),
                files.prefillVmfb.toString(), files.stepVmfb.toString(), files.paramsIrpa.toString(),
                files.vocabBin.toString(), files.decEmbedBin.toString(),
            ),
        )
        stream = s
        return Session(s).also { session = it }
    }

    @Synchronized
    override fun close() {
        session = null
        stream?.close(); stream = null
    }

    private inner class Session(private val s: IreeMoonshineStream) : StreamingAsrSession {
        @Synchronized override fun feed(pcm: FloatArray): String? = s.feedPcm(pcm)
        @Synchronized override fun finish(): String = s.finish().orEmpty()
        @Synchronized override fun reset() = s.reset()
        override fun close() = reset()
    }

    public companion object {
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
}
