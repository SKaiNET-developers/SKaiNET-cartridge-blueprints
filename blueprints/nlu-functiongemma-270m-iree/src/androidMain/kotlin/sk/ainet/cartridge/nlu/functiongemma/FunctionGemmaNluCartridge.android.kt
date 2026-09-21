package sk.ainet.cartridge.nlu.functiongemma

import android.content.Context
import android.util.Log
import kotlinx.io.files.Path
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.io.AndroidRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.transformers.iree.android.IreeKvSession
import sk.ainet.transformers.iree.android.IreeKvSpec
import java.io.File

/**
 * The Android cartridge: [FunctionGemmaEngine] bound to SKaiNET-transformers' released KV session
 * (`IreeKvSession`, `libskainet_iree_kv.so`) and GGUF tokenizer. Thread-safe: calls are serialized.
 *
 * @param cacheDir where the tokenized catalog prefix is cached between runs (an app's `filesDir` or `cacheDir`).
 * @param device IREE device; `null` = from the descriptor (`vulkan` when its target names an accelerator, else `local-task`).
 */
public class FunctionGemmaNluCartridge(
    pack: PackDir,
    cacheDir: File,
    config: FunctionGemmaEngine.Config = FunctionGemmaEngine.Config(),
    device: String? = null,
) : NluToolCallCartridge {

    private val engine = FunctionGemmaEngine(
        id = pack.id,
        catalog = ToolCatalog.load(pack.toolCatalog),
        config = config,
        openTokenizer = { ggufTokenizer(pack.tokenizerGguf.toString()) },
        openBackend = {
            IreeKvBackend(IreeKvSession(
                IreeKvSpec.functionGemma270m(config.chunk),
                device ?: if (pack.targetsAccelerator) IreeKvSession.VULKAN_DEVICE else "local-task",
                pack.graph("with-past").toString(), pack.parameters("with-past").toString(),
                pack.graph("prefill-with-past").toString(), pack.parameters("prefill-with-past").toString(),
                pack.graph("prefill-at").toString(), pack.parameters("prefill-at").toString(),
            ))
        },
        prefixCache = FilePrefixIdCache(Path(cacheDir.absolutePath)),
        cacheSalt = File(pack.tokenizerGguf.toString()).length().toString(16),
        log = { Log.i("FunctionGemmaNlu", it) },
    )

    override val id: String get() = engine.id
    override val toolNames: Set<String> get() = engine.toolNames
    public val catalog: ToolCatalog get() = engine.catalog
    public val warmUpTiming: Map<String, Long> get() = engine.warmUpTiming
    public val prefixTokens: Int get() = engine.prefixTokens

    @Synchronized override fun warmUp(): Unit = engine.warmUp()
    @Synchronized override fun resolve(transcript: String, budgetMs: Long): NluResolution = engine.resolve(transcript, budgetMs)
    @Synchronized override fun close(): Unit = engine.close()

    public companion object {
        /**
         * For apps that ship the pack_dir inside their APK assets: copies `assetPath/` to [into] once (IREE opens files
         * by path; the parameter archives are mmapped) and returns the [PackDir]. Files whose size already matches are
         * skipped. Apps that provision the pack_dir another way construct [PackDir] directly.
         */
        public fun packFromAssets(context: Context, assetPath: String, into: File = File(context.filesDir, assetPath)): PackDir {
            copyTree(context, assetPath, into)
            return PackDir(into.absolutePath)
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

        private fun ggufTokenizer(path: String): NluTokenizer {
            val fields = StreamingGGUFReader.open(AndroidRandomAccessSource.open(path)).use { it.fields }
            val tokenizer = TokenizerFactory.fromGgufFields(fields)
            return object : NluTokenizer {
                override fun encode(text: String): IntArray = tokenizer.encode(text)
                override fun decode(tokens: IntArray): String = tokenizer.decode(tokens)
            }
        }
    }
}

/** [KvBackend] over the released Android KV session. */
private class IreeKvBackend(private val session: IreeKvSession) : KvBackend {
    private class Snap(val inner: IreeKvSession.Snapshot) : KvBackend.Snapshot {
        override fun close() = inner.close()
    }

    override fun prefill(tokens: IntArray, n: Int): Int = session.prefill(tokens, n)
    override fun releasePrefill(): Unit = session.releasePrefill()
    override fun snapshot(): KvBackend.Snapshot = Snap(session.snapshot())
    override fun restore(snapshot: KvBackend.Snapshot): Unit = session.restore((snapshot as Snap).inner)
    override fun chunk(tokens: IntArray, n: Int): Int = session.chunk(tokens, n)
    override fun step(token: Int): Int = session.step(token)
    override fun close(): Unit = session.close()
}
