package sk.ainet.cartridge.nlu.qwen

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
 * The Android cartridge: [QwenEngine] bound to SKaiNET-transformers' released KV session (`IreeKvSession`,
 * `libskainet_iree_kv.so`) and GGUF tokenizer, for one flavor of the pack. The session's architecture (layers, heads,
 * key-value heads, RoPE base, mask heads) comes from the flavor's `manifest.json`, so nothing here is specific to one
 * checkpoint. Thread-safe: calls are serialized.
 *
 * @param flavor which flavor of the pack to run; `null` = the pack's only flavor.
 * @param cacheDir where the tokenized catalog prefix is cached between runs (an app's `filesDir` or `cacheDir`).
 * @param device IREE device; `null` = from the descriptor (`vulkan` when its target names an accelerator, else `local-task`).
 */
public class QwenNluCartridge(
    pack: PackDir,
    cacheDir: File,
    flavor: String? = null,
    config: QwenEngine.Config = QwenEngine.Config(),
    device: String? = null,
) : NluToolCallCartridge {

    public val flavor: String = flavor ?: pack.flavors.singleOrNull()
        ?: throw IllegalArgumentException("pack ${pack.id} bundles ${pack.flavors}; name the flavor to run")

    private val engine = QwenEngine(
        id = "${pack.id}-${this.flavor}",
        catalog = ToolCatalog.load(pack.toolCatalog),
        template = pack.chatTemplate(this.flavor),
        config = config,
        openTokenizer = { ggufTokenizer(pack.tokenizerGguf(this.flavor).toString()) },
        openBackend = {
            val f = this.flavor
            IreeKvBackend(IreeKvSession(
                IreeKvSpec.fromManifest(PackDir.readText(pack.kvManifest(f)), config.chunk),
                device ?: if (pack.targetsAccelerator) IreeKvSession.VULKAN_DEVICE else "local-task",
                pack.graph(f, "with-past").toString(), pack.parameters(f, "with-past").toString(),
                pack.graph(f, "prefill-with-past").toString(), pack.parameters(f, "prefill-with-past").toString(),
                pack.graph(f, "prefill-at").toString(), pack.parameters(f, "prefill-at").toString(),
                fnWithPast = "module.qwen_with_past",
                fnChunk = "module.qwen_prefill_with_past",
                fnPrefill = "module.qwen_prefill_at",
            ))
        },
        prefixCache = FilePrefixIdCache(Path(cacheDir.absolutePath)),
        cacheSalt = File(pack.tokenizerGguf(this.flavor).toString()).length().toString(16),
        log = { Log.i("QwenNlu", it) },
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
